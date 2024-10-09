"""
Triton Kernel
"""
import pytest
import torch
from torch import Tensor
import triton
import triton.language as tl
from typing import Tuple


def is_hip():
    return False


@triton.jit
def _attn_fwd_inner(
    acc: tl.tensor, 
    l_i: tl.tensor, 
    m_i: tl.tensor, 
    q: tl.tensor, 
    K_block_ptr: tl.tensor, 
    V_block_ptr: tl.tensor, 
    start_m: int, 
    qk_scale: float, 
    BLOCK_M: tl.constexpr, 
    HEAD_DIM: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    STAGE: tl.constexpr, 
    offs_m: tl.tensor, 
    offs_n: tl.tensor, 
    N_CTX: tl.constexpr
) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    """
    Compute a part of the attention mechanism for given query, key, and value tensors using blocks.
    
    Parameters:
    - acc (tl.tensor): Accumulator for the result.
    - l_i (tl.tensor): Log-sum of exponentials of the max-subtracted logits.
    - m_i (tl.tensor): Maximum logit observed so far (for stability in softmax).
    - q (tl.tensor): Loaded query block tensor.
    - K_block_ptr (tl.tensor): Pointer to the key tensor block.
    - V_block_ptr (tl.tensor): Pointer to the value tensor block.
    - start_m (int): Starting index of the block in query dimension.
    - qk_scale (float): Scaling factor for the dot products, typically the inverse square root of the dimensionality of the key vectors.
    - BLOCK_M, HEAD_DIM, BLOCK_N (tl.constexpr): Block size and head dimension constants.
    - STAGE (tl.constexpr): Indicates the computation stage.
    - offs_m, offs_n (tl.tensor): Offset tensors for matrix block indexing.
    - N_CTX (tl.constexpr): Total context size.
    
    Returns:
    - Tuple[tl.tensor, tl.tensor, tl.tensor]: Updated accumulator, log-sum of exponentials, and maximum logit tensors.
    """
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, N_CTX
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zeros
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.jit
def _moa_decode_split_fwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, Out: tl.tensor, L: tl.tensor, 
    Split_Index: tl.tensor, sm_scale: float,
    stride_qz: int, stride_qh: int, stride_qm: int, stride_qk: int,
    stride_kz: int, stride_khn: int, stride_kk: int,
    stride_vz: int, stride_vhn: int, stride_vn: int,
    stride_oz: int, stride_oh: int, stride_om: int, stride_on: int,
    stride_lz: int, stride_lhn: int, stride_lm: int,
    Z: int, H: int, N_Q: int, N_CTX_H: int,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KV_SPLIT_SIZE: tl.constexpr, STAGE: tl.constexpr
) -> None:
    """
    Perform the forward pass of the attention mechanism, processing in blocks for efficiency and applying auto-tuning.

    Parameters:
    - Q, K, V (tl.tensor): Query, Key, and Value tensors. For query, the input length should be 1
    - Split_Index (tl.tensor): The i-th number of split index is the head index of split i
    - sm_scale (float): Scale for softmax calculation.
    - M (tl.tensor): Tensor to store intermediate maximum values for softmax stability.
    - Out (tl.tensor): Output tensor.
    - stride_* (int): Strides for different dimensions of Q, K, V, and Out tensors.
    - Z, H, N_Q (int): Dimensions for batch, number of heads, query length.
    - N_CTX_H (int): Dimension for \sum_{h=0}^{H-1} N_CTX_h
    - HEAD_DIM, BLOCK_M, BLOCK_N (tl.constexpr): Dimension constants for head and block sizes.
    - KV_SPLIT_SIZE (tl.constexpr): Number of tokens for each key-value split.
    - STAGE (tl.constexpr): Current stage of the computation.

    Returns:
    - None: This function modifies the Out tensor in-place.
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    batch_id = tl.program_id(0)
    split_id = tl.program_id(1)

    head_id = tl.load(Split_Index + split_id).to(tl.int64)

    # pointer to batch
    q_offset = batch_id * stride_qz.to(tl.int64) + head_id * stride_qh.to(tl.int64)
    kv_offset = batch_id * stride_kz.to(tl.int64)
    o_offset = q_offset

    # pointer to vec
    l_offset = batch_id * stride_lz.to(tl.int64) + split_id * stride_lhn.to(tl.int64)

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(N_Q, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(KV_SPLIT_SIZE, HEAD_DIM),
        strides=(stride_vhn, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, KV_SPLIT_SIZE),
        strides=(stride_kk, stride_khn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_Q, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + l_offset,
        shape=(1, N_Q),
        strides=(stride_lhn, stride_lm),
        offsets=(0, 0),
        block_shape=(1, N_Q),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1, BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # skip boundary_check and do it in qk -> p

    # stage 1: off-band
    start_m = 0
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, KV_SPLIT_SIZE,
                                    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # write output
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0, 1))


class _mixture_of_sparse_attention_decode(torch.autograd.Function):
    """
    # During Decode
    input:
        q: (Z, H, N_IN, L)
        k, v: (Z, \sum_i^H N_CTX, L)
        sm_scale: float
        head_index: (H+1)
    output:
        o: (Z, H, N_IN, L)
    """

    @staticmethod
    def forward(
        ctx, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        head_index: Tensor, 
        sm_scale: float, 
        causal: bool,
    ):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        
        BATCH_SIZE = q.shape[0]
        NUM_HEAD = q.shape[1]
        QUERY_SIZE = q.shape[2]
        HEAD_DIM = q.shape[3]
        assert HEAD_DIM in {16, 32, 64, 128, 256}


        KV_SPLIT_SIZE = 256 # 16 * 16
        CTX_HEAD_SIZE = k.shape[1]
        KV_SPLIT_NUM = triton.cdiv(CTX_HEAD_SIZE, KV_SPLIT_SIZE) # noqa: assume each head can be divided by KV_SPLIT_SIZE
        

        # prepare output
        o = torch.empty_like(q)
        l = torch.empty((BATCH_SIZE, KV_SPLIT_NUM, QUERY_SIZE), dtype=q.dtype, device=q.device)

        # prepare index
        split_index = head_index_to_split_index(head_index, KV_SPLIT_SIZE)

        # stage = 3 if causal else 1
        stage = 1 # noqa: multiply with all inputs
        extra_kern_args = {}

        # get grid parameters
        grid = (BATCH_SIZE, KV_SPLIT_NUM, 1)
        _moa_decode_split_fwd[grid](
            q, k, v, o, l, #
            split_index, sm_scale,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(-1),  #
            v.stride(0), v.stride(1), v.stride(-1),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            l.stride(0), l.stride(1), l.stride(2),
            BATCH_SIZE, NUM_HEAD, QUERY_SIZE, CTX_HEAD_SIZE, 
            HEAD_DIM, BLOCK_M=16, BLOCK_N=16, KV_SPLIT_SIZE=KV_SPLIT_SIZE, STAGE=stage,  #
            **extra_kern_args,
        )
        
        return o

def head_index_to_split_index(head_index: torch.Tensor, split_size: int) -> Tensor:
    """
    Convert head index to split index.

    Parameters:
    - head_index (torch.Tensor): Head index tensor, shape (num_head+1). The i-th value is the start index of the i-th head.
    - split_size (int): Size of each split.

    Returns:
    - Tensor: Split to head index tensor, shape (num_split).  The i-th number of split index is the index of the head the split is in.
    """
    # The total length of the sequence based on the last index in head_index
    total_length = head_index[-1] - head_index[0]

    # Calculate the number of splits
    num_splits = (total_length + split_size - 1) // split_size

    # Initialize the tensor that will hold the split to head index mapping
    split_to_head_index = torch.empty(num_splits, dtype=torch.long)

    # Iterate through each head
    num_heads = len(head_index) - 1
    for i in range(num_heads):
        # Start and end of the current head
        start_idx = head_index[i]
        end_idx = head_index[i + 1] - 1

        # Determine the start and end splits covered by this head
        start_split = start_idx // split_size
        end_split = end_idx // split_size

        # Assign head index to all splits in the range
        split_to_head_index[start_split:end_split + 1] = i

    return split_to_head_index
    


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 16, 1, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [True]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["triton-fp16"] +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="ms",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "Q_LEN": 1,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, Q_LEN, HEAD_DIM, causal, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, Q_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
