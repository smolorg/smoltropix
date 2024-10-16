from typing import NamedTuple

import mlx
import mlx.nn
import mlx.core as mx


class KVCache(NamedTuple):
    k: mx.array
    v: mx.array

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        return cls(
            k = mx.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=mx.bfloat16),
            v = mx.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=mx.bfloat16),
        )

    def update(self, xk: mx.array, xv: mx.array, layer_idx: int, cur_pos: int, n_rep: int):
        """
        Updates the cache with new key and value tensors.

        Args:
            xk (mx.array): New key tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            xv (mx.array): New value tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            layer_idx (int): The index of the layer to update.
            cur_pos (int): The current position in the sequence to start inserting.
            n_rep (int): The number of times to repeat the keys and values along the sequence dimension.

        Returns:
            Tuple[mx.array, mx.array, KVCache]:
                - keys: Updated or repeated keys tensor.
                - values: Updated or repeated values tensor.
        """
        xk = xk.astype(self.k.dtype)
        xv = xv.astype(self.v.dtype)

        insert_len = xk.shape[1]
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xv

        if cur_pos == 0:
            # If inserting at the beginning, repeat the new keys and values
            keys = mx.repeat(xk, n_rep, axis=2)
            values = mx.repeat(xv, n_rep, axis=2)
        else:
            # Otherwise, repeat the existing keys and values from the cache
            keys = mx.repeat(self.k[layer_idx], n_rep, axis=2)
            values = mx.repeat(self.v[layer_idx], n_rep, axis=2)

        return keys, values, self
