"""
Run tests for the mixture_of_sparse_attention function.

`python -m unittest tests.mixture_of_attention.TestMixtureOfSparseAttention.test_decode_stage`
"""

import unittest
import torch
from MoA.kernels.mixture_of_attention import mixture_of_sparse_attention  # Adjust import according to your project structure

class TestMixtureOfSparseAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Determine if CUDA is available and set the default tensor type to CUDA if possible
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Tests will run on: {cls.device}")
        # Set the test implementations
        cls.gen_data_func = torch.rand
        # cls.gen_data_func = torch.ones
        cls.implementations = ["sdpa", "moa"]

    def buld_head_index(self, num_head_for_each_group, cache_size_for_each_group):
        num_group = len(num_head_for_each_group)
        current_index = 0
        head_index = [current_index]
        for group_id in range(num_group):
            num_heads = num_head_for_each_group[group_id]
            cache_size = cache_size_for_each_group[group_id]
            for i in range(num_heads):
                current_index += cache_size
                head_index.append(current_index)
        return torch.tensor(head_index, dtype=torch.int64, device=self.device).contiguous()


    def test_prefill_stage(self):
        """Test the prefill stage of the mixture_of_sparse_attention function with different implementations."""
        bsz, num_heads, seq_len, hidden_dim = 2, 4, 128, 64
        sm_scale = 0.1
        attention_dropout = 0.0

        q = self.gen_data_func(bsz, num_heads, seq_len, hidden_dim, dtype=torch.float16, device=self.device)
        k = self.gen_data_func(bsz, num_heads, seq_len, hidden_dim, dtype=torch.float16, device=self.device)
        v = self.gen_data_func(bsz, num_heads, seq_len, hidden_dim, dtype=torch.float16, device=self.device)

        num_head_for_each_group = [2, 2]
        assert num_heads == sum(num_head_for_each_group)
        cache_size_for_each_group = [128, 128]
        head_index = self.buld_head_index(num_head_for_each_group, cache_size_for_each_group)

        global_cache_size_uniform = 64
        global_cache_size = torch.tensor([global_cache_size_uniform for h in range(num_heads)], dtype=torch.int64, device=self.device).contiguous()
        
        local_cache_size = []
        for group_id, num_head in enumerate(num_head_for_each_group):
            for head_id in range(num_head):
                local_cache_size.append(cache_size_for_each_group[group_id] - global_cache_size_uniform)
        local_cache_size = torch.tensor(local_cache_size, dtype=torch.int64, device=self.device).contiguous()
        
        output = dict()
        for implementation in self.implementations:
            print(f"Testing prefill with implementation: {implementation}")
            output[implementation] = mixture_of_sparse_attention(
                q, k, v, sm_scale, attention_dropout=attention_dropout, implementation=implementation, local_size=local_cache_size, sink_size=global_cache_size, head_index=head_index,
            )
            # Check shape
            self.assertEqual(output[implementation].shape, (bsz, seq_len, num_heads, hidden_dim))

        # Assert equality across different implementations
        torch.testing.assert_close(output[self.implementations[0]], output[self.implementations[1]], rtol=1e-3, atol=1e-3)


    def test_decode_stage(self):
        """Test the decode stage of the mixture_of_sparse_attention function."""
        bsz, q_len, hidden_dim = 2, 1, 16  # Assuming q_len = 1 for decode stage

        # num_head_for_each_group = [8, 4]
        # cache_size_for_each_group = [64, 128]
        num_head_for_each_group = [1,]
        cache_size_for_each_group = [64,]

        num_group = len(num_head_for_each_group)

        current_index = 0
        head_index = [current_index]
        ks = []
        vs = []

        for group_id in range(num_group):
            num_heads = num_head_for_each_group[group_id]
            cache_size = cache_size_for_each_group[group_id]

            for i in range(num_heads):
                current_index += cache_size
                head_index.append(current_index)

            k = self.gen_data_func(bsz, num_heads, cache_size, hidden_dim, dtype=torch.float16, device=self.device)
            v = self.gen_data_func(bsz, num_heads, cache_size, hidden_dim, dtype=torch.float16, device=self.device)

            # k = torch.arange(cache_size, dtype=torch.float16, device=self.device)[None, None, :, None].expand(bsz, num_heads, -1, hidden_dim)
            # v = torch.arange(cache_size, dtype=torch.float16, device=self.device)[None, None, :, None].expand(bsz, num_heads, -1, hidden_dim)

            ks.append(k.reshape(bsz, num_heads * cache_size, hidden_dim))
            vs.append(v.reshape(bsz, num_heads * cache_size, hidden_dim))

        num_heads = sum(num_head_for_each_group)
        q = self.gen_data_func(bsz, num_heads, q_len, hidden_dim, dtype=torch.float16, device=self.device).contiguous()
        k = torch.cat(ks, dim=1).contiguous()
        v = torch.cat(vs, dim=1).contiguous()
        head_index = torch.tensor(head_index, dtype=torch.int64, device=self.device).contiguous()
        sm_scale = sum(num_head_for_each_group)**-0.5

        output = dict()
        for implementation in self.implementations:
            print("implement with ", implementation)
            output[implementation] = mixture_of_sparse_attention(
                q, k, v, sm_scale, head_index=head_index, attention_mask=None, attention_dropout=0.0, implementation=implementation,
            )
            # check shape
            self.assertEqual(output[implementation].shape, (bsz, q_len, num_heads, hidden_dim))

        # Assert equality
        torch.testing.assert_close(output[self.implementations[0]], output[self.implementations[1]], rtol=1e-3, atol=1e-3)

    def test_invalid_input(self):
        """Test function behavior with invalid input shapes."""
        bsz, num_heads, seq_len, hidden_dim = 2, 4, 10, 64
        sm_scale = 0.1
        q = self.gen_data_func(bsz, num_heads, seq_len, hidden_dim, dtype=torch.float16, device=self.device)
        k = self.gen_data_func(bsz, num_heads, seq_len, hidden_dim + 1, dtype=torch.float16, device=self.device)  # Invalid shape
        v = self.gen_data_func(bsz, num_heads, seq_len, hidden_dim, dtype=torch.float16, device=self.device)

        with self.assertRaises(AssertionError):
            mixture_of_sparse_attention(q, k, v, sm_scale)

if __name__ == '__main__':
    unittest.main(defaultTest='TestMixtureOfSparseAttention.test_decode_stage')

