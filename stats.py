from typing import NamedTuple

import mlx
import mlx.nn
import mlx.core as mx


class AttnStats(NamedTuple):
    entropy: mx.array   # (bsz, n_layers, n_heads)
    varentropy: mx.array    # (bsz, n_layers, n_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=mx.zeros((bsz, n_layers, n_heads), dtype=mx.float32),
            varentropy=mx.zeros((bsz, n_layers, n_heads), dtype=mx.float32),
            n_layers=n_layers,
            n_heads=n_heads
        )
    
    @property
    def avg_entropy(self):
        return self.entropy.sum(axis=-1, keepdims=False)    # avg across heads
    
    @property
    def std_error(self):
        return mx.sqrt(mx.mean(self.varentropy)) / (self.n_layers * self.n_heads)
    
    def update(self, scores: mx.array, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = mlx.nn.softmax(scores, axis=-1)
        new_entropy = -mx.sum(mx.where(probs > 0, probs * mx.log(probs), 0), axis=-1)
        new_varentropy = mx.sum(probs * (mx.log(probs) + mx.expand_dims(new_entropy, -1))**2, axis=-1)

        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy
    
        return self
