"""
Fused Attention
===============

This is a Triton implementation of the prefill attention kernel. It support block sparse.

"""
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

"""
Fused Prefill Kernel.
support SemSA layout.
"""

@triton.jit
def _sparse_attention_prefill_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    lut,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lx,
    Z, H, N_CTX,
    NNZ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) # block id x
    off_hz = tl.program_id(1) # block id y
    lut_indicator = tl.program_id(1) % H
    qvk_offset = off_hz * stride_qh
    lut_offset = lut_indicator * stride_lz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # skip boundary_check and do it in qk -> p
    # q = tl.load(Q_block_ptr) # shape (BLOCK_M, BLOCK_DMODEL)
    q = (q * qk_scale).to(tl.float16)

    # last_nnz_id: used when padding
    last_nnz_id = -1

    # Deprecated, slow down the process
    # for nnz_id in range(NNZ):
    #     # we use nnz_id to indicate which block will be computed
    #     present_nnz_id = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
    #     start_n = present_nnz_id * BLOCK_N
    #     # skip those nnz that exceed N_CTX
    #     if start_n < N_CTX:
    #         VALID_NNZ += 1

    # loop over k, v and update accumulator
    for nnz_id in range(NNZ):
        # we use nnz_id to indicate which block will be computed
        present_nnz_id = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
        start_n = present_nnz_id * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N) # hint for compiler
        present_nnz_id = present_nnz_id.to(tl.int32)
        # TODO: not supported by Triton yet
        # tl.device_print("start_n: ", start_n)
        # if start_n >= N_CTX:
            # break   

        # -- compute qk ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)), boundary_check=(0, 1), padding_option="zero") # skip boundary_check and do it in qk -> p
        # k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf")) # mask the upper triangular part
        qk = tl.where((offs_m[:, None] < N_CTX) & ((start_n + offs_n)[None, :] < N_CTX), qk, float("-inf")) # mask the part that exceed N_CTX

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)

        p = tl.math.exp2(qk - m_ij[:, None])
        # if a blcok has a row filled with "inf" (e.g., in the upper triangular), then m_ij == '-inf'. In this case, p should be set to '0', or exp2('inf') will trigger NaN
        p = tl.where(m_ij[:, None] == tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32), 0.0, tl.math.exp2(qk - m_ij[:, None])) # ignore the blocks in upper triangular part, which is indicated by last_nnz_id
        p = p * (last_nnz_id!=present_nnz_id) # ignore the blocks in upper triangular part, which is indicated by last_nnz_id

        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i *= alpha
        l_i_new = l_i + beta * l_ij

        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        # scale acc
        acc_scale = l_i / l_i_new
        acc = acc * acc_scale[:, None]

        # update acc
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)), boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zero
        p = p.to(tl.float16)
        acc += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

        # update last_nnz_id
        last_nnz_id = present_nnz_id

    # write back l and m
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, l_i)
    # tl.store(m_ptrs, m_i)

    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

class _sparse_attention_prefill(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, lut, BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
        """
        input:
            q, k, v: (Z, H, N_CTX, L)
            sm_scale: float
            lut: (H, N_CTX/BLOCK_M, nnz)
            BLOCK_M, BLOCK_N: int
        output:
            o: (Z, H, N_CTX, L)
        """
        dtype = q.dtype
        assert dtype == torch.float16

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}

        # Deprecated, slow down the process. We do it inside the kernel.
        # pad the input so that the N_CTX is the multiply of BLOCK_N
        # ------------------------------- #
        # def pad_tensor(tensor, block_size, pad_value=0):
        #     _, _, length, _ = tensor.shape
        #     pad_size = (block_size - length % block_size) % block_size
        #     if pad_size > 0:
        #         padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), "constant", pad_value)
        #     else:
        #         padded_tensor = tensor
        #     return padded_tensor, pad_size

        # q, Q_pad_size = pad_tensor(q, BLOCK_N, 0)
        # k, K_pad_size = pad_tensor(k, BLOCK_N, 0)  # Similar for K with BLOCK_N
        # v, V_pad_size = pad_tensor(v, BLOCK_N, 0)  # Similar for V with BLOCK_M
        # ------------------------------- #

        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        # the maximum number of non-zero blocks will not exceed N_CTX / BLOCK_N
        # NNZ_LUT = lut.shape[-1]
        NNZ = min(lut.shape[-1], math.ceil(q.shape[2] / BLOCK_N))

        num_warps = 4 if Lk <= 64 else 8
        num_stages = 4 if BLOCK_M <= 32 else 2

        # ------------------------------- #
        _sparse_attention_prefill_fwd_kernel[grid](
            q, k ,v, sm_scale,
            o,
            lut,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lut.stride(0), lut.stride(1), lut.stride(2),
            q.shape[0], q.shape[1], q.shape[2], NNZ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N, 
            num_warps=num_warps,
            num_stages=num_stages)
        # ------------------------------- #

        # after computation, remove padding from the output
        # o = o[..., :-Q_pad_size, :] if Q_pad_size > 0 else o
        # ------------------------------- #
        return o

sparse_attention_prefill = _sparse_attention_prefill.apply

class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx, q, k, v, sm_scale, lut, BLOCK_M: int = 64, BLOCK_N: int = 64,
            attention_mask = None, attention_dropout = 0.0
        ) -> torch.Tensor:
        """
        input:
            q, k, v: (Z, H, N_CTX, L)
            sm_scale: float
            lut: (H, N_CTX/BLOCK_M, nnz)
            BLOCK_M, BLOCK_N: int
        output:
            o: (Z, H, N_CTX, L)
        """
        dtype = q.dtype
        assert dtype == torch.float16

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}

        bsz, num_heads, q_len, _ = q.shape
        kv_seq_len = k.size(2)

        _is_decode = q_len < kv_seq_len
        # decode
        if _is_decode:
            assert q_len == 1

            query_states = q
            key_states = k
            value_states = v

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                attention_dropout,
                is_causal=attention_mask is None and q_len > 1,
                scale=sm_scale,
            )

            return attn_output

        # prefill
        else:
            return F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attention_mask,
                dropout_p=attention_dropout,
                is_causal=attention_mask is None and q_len > 1,
                scale=sm_scale,
            )
            # return sparse_attention_prefill(q, k, v, sm_scale, lut, BLOCK_M, BLOCK_N)
            # If lut is replaced by (sink, local), then:
            # (sink, local) = lut
            # returen sparse_attention_wo_lut_prefill(q, k, v, sm_scale, sink, local, BLOCK_M, BLOCK_N)

sparse_attention = _sparse_attention.apply

"""
Fused Decode Kernel.
NOT fully implemented yet.
"""

@triton.jit
def _sparse_attention_decode_fwd_kernel(
    Q, K, V, sm_scale,
    Out, L, M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    stride_luth, NNZ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    lut,
):
    # parallelize over z, h and k;
    # we use different grid for each z, h and block of k
    # grid is of shape (K // BLOCK_N, Z * H, 1)

    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    # assert stride_kh == stride_vh

    # the n_th nnz block from the lut
    start_nnz = tl.load(lut + off_hz * stride_luth + start_n)

    kv_offset = off_hz * stride_kh
    q_offset = off_hz * stride_qh
    o_offset = off_hz * stride_oh
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(1, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(1, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_nnz * BLOCK_N),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(start_nnz * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(NNZ, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_n, 0),
        block_shape=(1, BLOCK_DMODEL),
        order=(1, 0)
    )
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    
    # use k, v to update accumulator
    # -- compute qk ----
    k = tl.load(K_block_ptr)
    qk = tl.expand_dims(tl.sum((tl.trans(q) * k), axis=0), axis=0).to(tl.float32) # qk = tl.dot(q, k), yet tl.dot do not support matrix with smaller dimension than 16
    # if N_CTX is not a multiple of BLOCK_N, some qk is not valid
    # TODO: this may slow down the kernel a little. Skip the process if N_CTX is guaranteed to be a multiple of BLOCK_N
    k_indices = start_nnz * BLOCK_N + tl.arange(0, BLOCK_N)
    qk = tl.where(k_indices < N_CTX, qk, float("-inf")) # for the last block, some qk is not valid if N_CTX is not a multiple of BLOCK_N

    # -- compute m, p, l ----
    # in some cases, use const m is better than max m to avoid redundant store. e.g. m = 10.0
    m = tl.max(qk, axis=1, return_indices=False)
    p = tl.math.exp2(qk - m)
    l = tl.sum(p, axis=1)
    # update acc
    v = tl.load(V_block_ptr).to(tl.float16)
    p = p.to(tl.float16)
    acc = tl.expand_dims(tl.sum((tl.trans(p) * v), axis=0), axis=0) # acc = tl.dot(p, v)

    # -- store o, l, m ----
    tl.store(O_block_ptr, acc)
    l_ptrs = L + off_hz * NNZ + start_n + tl.arange(0, 1)
    m_ptrs = M + off_hz * NNZ + start_n + tl.arange(0, 1)
    tl.store(l_ptrs, l)
    tl.store(m_ptrs, m)

class _sparse_attention_decode(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, sm_scale: float, lut: Tensor, BLOCK_N: int = 64) -> torch.Tensor:
        """
        input:
            q: (Z, H, 1, L)
            k: (Z, H, N_CTX, L)
            v: (Z, H, N_CTX, L)
            sm_scale: float
            lut: (Z, H, nnz)
        output:
            o: (Z, H, 1, L)
        """
        dtype = q.dtype
        assert dtype == torch.float16

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}

        Tc = lut.shape[-1]
        z = q.shape[0]
        h = q.shape[1]
        n_ctx = k.shape[2]

        o = torch.zeros(z, h, Tc, k.shape[3], dtype=q.dtype, device=q.device)

        l = torch.zeros((z * h, Tc), dtype=torch.float32, device=q.device)
        m = torch.zeros((z * h, Tc), dtype=torch.float32, device=q.device)

        grid = (Tc, z * h, 1)

        # ------------------------------- #
        _sparse_attention_decode_fwd_kernel[grid](
            q, k, v, sm_scale,
            o, l, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            z, h, n_ctx,
            lut.stride(1), Tc,
            BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N,
            lut=lut
        )
        l = l.view(z, h, Tc, 1)
        mm = torch.max(m, dim=-1)[0]
        m = torch.exp2(m-mm[:,None]).view(z, h, 1, Tc)
        o = (torch.matmul(m.half(), o) / torch.matmul(m, l)).to(dtype)
        # ------------------------------- #

        # ctx.save_for_backward(q, k, v, o, L, m)
        # ctx.grid = grid
        # ctx.sm_scale = sm_scale
        # ctx.BLOCK_DMODEL = Lk
        # ctx.causal = causal
        return o

# sparse_attention_decode = _sparse_attention_decode.apply



@triton.jit
def _sparse_attention_prefill_fwd_kernel_wo_lut(
    Q, K, V, sm_scale,
    Out,
    sink, # [H,]
    local, # [H,]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) # block id x
    off_hz = tl.program_id(1) # block id y
    head_id = tl.program_id(1) % H
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # skip boundary_check and do it in qk -> p
    # q = tl.load(Q_block_ptr) # shape (BLOCK_M, BLOCK_DMODEL)
    q = (q * qk_scale).to(tl.float16)

    # last_nnz_id: used when padding
    last_nnz_id = -1

    # Deprecated, slow down the process
    # for nnz_id in range(NNZ):
    #     # we use nnz_id to indicate which block will be computed
    #     present_nnz_id = tl.load(lut + lut_offset + start_m * stride_lh + nnz_id * stride_lx)
    #     start_n = present_nnz_id * BLOCK_N
    #     # skip those nnz that exceed N_CTX
    #     if start_n < N_CTX:
    #         VALID_NNZ += 1
    num_sink = tl.load(sink + head_id).to(tl.int32)
    num_local = tl.load(local + head_id).to(tl.int32)
    num_blocks = N_CTX / BLOCK_N
    
    
    present_nnz_id = 0
    # loop over k, v and update accumulator
    while(present_nnz_id < num_blocks):
        min_local = tl.max(0, present_nnz_id + 1 - num_local)
        max_sink = tl.min(num_sink - 1, present_nnz_id)
        if (present_nnz_id > max_sink and present_nnz_id < min_local):
            continue
        # we use nnz_id to indicate which block will be computed
        start_n = present_nnz_id * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N) # hint for compiler
        present_nnz_id = present_nnz_id.to(tl.int32)
        # -- compute qk ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)), boundary_check=(0, 1), padding_option="zero") # skip boundary_check and do it in qk -> p
        # k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf")) # mask the upper triangular part
        qk = tl.where((offs_m[:, None] < N_CTX) & ((start_n + offs_n)[None, :] < N_CTX), qk, float("-inf")) # mask the part that exceed N_CTX

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)

        p = tl.math.exp2(qk - m_ij[:, None])
        # if a blcok has a row filled with "inf" (e.g., in the upper triangular), then m_ij == '-inf'. In this case, p should be set to '0', or exp2('inf') will trigger NaN
        p = tl.where(m_ij[:, None] == tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32), 0.0, tl.math.exp2(qk - m_ij[:, None])) # ignore the blocks in upper triangular part, which is indicated by last_nnz_id
        p = p * (last_nnz_id!=present_nnz_id) # ignore the blocks in upper triangular part, which is indicated by last_nnz_id

        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i *= alpha
        l_i_new = l_i + beta * l_ij

        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        # scale acc
        acc_scale = l_i / l_i_new
        acc = acc * acc_scale[:, None]

        # update acc
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)), boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zero
        p = p.to(tl.float16)
        acc += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

        # update last_nnz_id
        last_nnz_id = present_nnz_id
        present_nnz_id += 1

    # write back l and m
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, l_i)
    # tl.store(m_ptrs, m_i)

    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

class _sparse_attention_wo_lut_prefill(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, sink, local, BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
        """
        input:
            q, k, v: (Z, H, N_CTX, L)
            sm_scale: float
            sink: (H,)
            local: (H,)
            BLOCK_M, BLOCK_N: int
        output:
            o: (Z, H, N_CTX, L)
        """
        dtype = q.dtype
        assert dtype == torch.float16

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}

        # Deprecated, slow down the process. We do it inside the kernel.
        # pad the input so that the N_CTX is the multiply of BLOCK_N
        # ------------------------------- #
        # def pad_tensor(tensor, block_size, pad_value=0):
        #     _, _, length, _ = tensor.shape
        #     pad_size = (block_size - length % block_size) % block_size
        #     if pad_size > 0:
        #         padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), "constant", pad_value)
        #     else:
        #         padded_tensor = tensor
        #     return padded_tensor, pad_size

        # q, Q_pad_size = pad_tensor(q, BLOCK_N, 0)
        # k, K_pad_size = pad_tensor(k, BLOCK_N, 0)  # Similar for K with BLOCK_N
        # v, V_pad_size = pad_tensor(v, BLOCK_N, 0)  # Similar for V with BLOCK_M
        # ------------------------------- #

        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        num_warps = 4 if Lk <= 64 else 8
        num_stages = 4 if BLOCK_M <= 32 else 2

        # ------------------------------- #
        _sparse_attention_prefill_fwd_kernel_wo_lut[grid](
            q, k ,v, sm_scale,
            o,
            sink,
            local,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N, 
            num_warps=num_warps,
            num_stages=num_stages)
        # ------------------------------- #

        # after computation, remove padding from the output
        # o = o[..., :-Q_pad_size, :] if Q_pad_size > 0 else o
        # ------------------------------- #
        return o