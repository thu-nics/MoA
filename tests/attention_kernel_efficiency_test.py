import torch
import unittest
import math
import timeit

from MoA.kernels.mixture_of_attention import mixture_of_sparse_attention

def pytorch_attention(q, k, v, causal_mask):
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    causal_mask = torch.triu(torch.ones_like(attn_scores), diagonal=1) * float('-inf')
    attn_scores += causal_mask
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
    o = torch.matmul(attn_probs, v)
    return o

def moa_attention(q, k, v, sink_size, local_size):
    scale_factor = 1 / math.sqrt(q.size(-1))
    return mixture_of_sparse_attention(q, k, v, sm_scale = scale_factor, sink_size = sink_size, local_size = local_size)

def scaled_dot_product_attention(q, k, v, causal_mask):
    # This is a placeholder as scaled dot product attention is not a direct part of PyTorch
    o = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True,
    )
    return o

class AttentionRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bsz, cls.num_heads, cls.seq_len, cls.hidden_dim = 1, 1, 4096, 128
        cls.device = "cuda"

        print(cls)

        # Tensors on GPU in float16
        cls.q = torch.rand(cls.bsz, cls.num_heads, cls.seq_len, cls.hidden_dim, dtype=torch.float16, device=cls.device)
        cls.k = torch.rand(cls.bsz, cls.num_heads, cls.seq_len, cls.hidden_dim, dtype=torch.float16, device=cls.device)
        cls.v = torch.rand(cls.bsz, cls.num_heads, cls.seq_len, cls.hidden_dim, dtype=torch.float16, device=cls.device)
        cls.causal_mask = torch.tril(torch.ones(cls.seq_len, cls.seq_len, dtype=torch.bool, device=cls.device), diagonal=0)

    def test_pytorch_attention(self):
        elapsed_time = timeit.timeit(lambda: pytorch_attention(self.q, self.k, self.v, self.causal_mask), number=10)
        print(f"PyTorch Attention time: {elapsed_time} seconds")

    def test_moa_attentions(self):
        block_size = 64
        sink_size = torch.tensor([1 for _ in range(self.num_heads)], device=self.device)
        density = 0.2
        local_block_num = int(self.seq_len * density / block_size) - 1
        local_size = torch.tensor([local_block_num for _ in range(self.num_heads)], device=self.device)
        print(f"local size: {local_block_num}")

        elapsed_time = timeit.timeit(lambda: moa_attention(self.q, self.k, self.v, sink_size, local_size), number=10)
        print(f"Mixture of Attentions time: {elapsed_time} seconds")

    def test_sdpa_attention(self):
        elapsed_time = timeit.timeit(lambda: scaled_dot_product_attention(self.q, self.k, self.v, self.causal_mask), number=10)
        print(f"Scaled Dot Product Attention time: {elapsed_time} seconds")

if __name__ == "__main__":
    unittest.main()