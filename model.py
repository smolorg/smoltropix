import math
import struct
from typing import Optional, Tuple

import mlx
import mlx.nn
import mlx.core as mx

from config import ModelParams
from kvcache import KVCache
from stats import AttnStats
from weights import XfmrWeights, LayerWeights
from utils import complexarray


float32_max = struct.unpack('f', struct.pack('I', 0x7f7fffff))[0]
DEFAULT_MAX_VALUE = -0.7 * float32_max


@mx.compile
def rms_norm(x: mx.array, w: mx.array, eps: float = 1e-10) -> mx.array:
    return mx.fast.rms_norm(x, w, eps)


def apply_rotary_emb(xq: mx.array, xk: mx.array, freqs_cis: complexarray, dtype: mx.Dtype = mx.float32) -> Tuple[mx.array, mx.array]:
    reshape_xq = xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(mx.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_ = complexarray(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = complexarray(reshape_xk[..., 0], reshape_xk[..., 1])
    fc_expanded = freqs_cis.expand_dims(0).expand_dims(2)
    xq_out = xq_ * fc_expanded
    xk_out = xk_ * fc_expanded
    xq_out = mx.stack([xq_out.real, xq_out.imag], axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = mx.stack([xk_out.real, xk_out.imag], axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(dtype), xk_out.astype(dtype)


def attention(x: mx.array, layer_weights: LayerWeights, model_params: ModelParams,
              cur_pos: int, layer_idx: int, freqs_cis: complexarray, kvcache: KVCache,
              attn_mask: Optional[mx.array] = None) -> Tuple[mx.array, KVCache, mx.array]:
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = mx.matmul(x, layer_weights.wq.T).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = mx.matmul(x, layer_weights.wk.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = mx.matmul(x, layer_weights.wv.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = mx.transpose(xq, (0, 2, 1, 3))     # (bs, n_heads, seqlen, head_dim)
    keys = mx.transpose(keys, (0, 2, 3, 1))     # (bs, n_heads, head_dim, cache_len + seqlen)
    values = mx.transpose(values, (0, 2, 1, 3))     # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = mx.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.astype(mx.float32)  # always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = mx.where(scores != 0.0, scores, DEFAULT_MAX_VALUE)
    padded_logits = mx.where((mask >= DEFAULT_MAX_VALUE * 0.5), scores, DEFAULT_MAX_VALUE)
    scores = mx.softmax(padded_logits, axis=-1).astype(x.dtype)
    output = mx.matmul(scores, values)
    output = mx.swapaxes(output, 1, 2).reshape(xq.shape[0], xq.shape[2], -1)
    out = mx.matmul(output, layer_weights.wo.T)
    return out, kvcache, pre_scores


def feed_forward(x: mx.array, layer_weights: LayerWeights) -> mx.array:
    return mx.matmul(mlx.nn.silu(mx.matmul(x, layer_weights.w1.T)) * mx.matmul(x, layer_weights.w3.T), layer_weights.w2.T)


def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: mx.array, 
         cur_pos: int, freqs_cis: complexarray, kvcache: KVCache, 
         attn_mask: Optional[mx.array] = None) -> Tuple[mx.array, KVCache, mx.array, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = mx.matmul(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)
    return logits, kvcache, scores, attn_stats
