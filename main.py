import math
import rich
from pathlib import Path

import mlx
import mlx.core as mx
import tyro

from config import LLAMA_1B_PARAMS, ModelParams
from kvcache import KVCache
from model import xfmr, DEFAULT_MAX_VALUE
from sampler import SamplerConfig, sample
from prompts import PROMPT_TEMPLATE
from sampler import sample, isin
from tokenizer import Tokenizer
from weights import load_weights, XfmrWeights
from utils import complexarray, COLORS


DEFAULT_WEIGHTS_PATH = Path(__file__).parent / 'weights'


def apply_scaling(freqs: mx.array):
    SCALE_FACTOR = 8
    LOW_FREQ_FACTOR = 1
    HIGH_FREQ_FACTOR = 4
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR
    wavelens = 2 * math.pi / freqs

    freqs = mx.where(wavelens > low_freq_wavelen, freqs / SCALE_FACTOR, freqs)
    is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
    smooth_factors = (OLD_CONTEXT_LEN / wavelens - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
    smooth_freqs = (1 - smooth_factors) * freqs / SCALE_FACTOR + smooth_factors * freqs
    freqs = mx.where(is_medium_freq, smooth_freqs, freqs)
    return freqs


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: mx.Dtype = mx.float32) -> complexarray:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)
    t = mx.arange(end, dtype=dtype)
    freqs = mx.outer(t, freqs)
    return complexarray(mx.zeros_like(freqs), freqs).exp()


def build_attn_mask(seqlen: int, start_pos: int) -> mx.array:
    mask = mx.zeros((seqlen, seqlen), dtype=mx.float32)
    if seqlen > 1:
        mask = mx.ones((seqlen, seqlen)) * float('-inf')
        mask = mx.triu(mask, k=1)
        mask = mx.concatenate([mx.zeros((seqlen, start_pos)), mask], axis=1)
    return mask


def main(input: str, weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath("1B-Instruct")):
    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights(weights_path.absolute())
    tokenizer = Tokenizer("./tokenizer.model")

    def generate(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens):
        gen_tokens = None
        cur_pos = 0
        tokens = mx.array([tokens], dtype=mx.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
        kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
        logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        next_token = mx.argmax(logits[:, -1], axis=-1, keepdims=True).astype(mx.int32)
        gen_tokens = next_token
        rich.print(f"[{COLORS['lelv']}]{tokenizer.decode([next_token.item()])}[/{COLORS['lelv']}]", end='', flush=True)
        cur_pos = seqlen
        stop = mx.array([128001, 128008, 128009])
        sampler_cfg = SamplerConfig()
        while cur_pos < 8192:
            cur_pos += 1
            logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
            mask = (mx.arange(scores.shape[-1]) >= cur_pos)
            mask = mask.reshape(1, 1, 1, -1)
            scores = mx.where(mask, DEFAULT_MAX_VALUE, scores)
            next_token, color, metrics = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
            gen_tokens = mx.concatenate((gen_tokens, next_token))
            decoded = tokenizer.decode(next_token.tolist()[0])
            if color != "nocolor":
                rich.print(f"[{color}]{decoded}[/{color}]", end='', flush=True)
            else:
                print(f"{decoded}", end='', flush=True)
            if isin(next_token, stop).any():
                break

    prompt = PROMPT_TEMPLATE.format(user_input=input)
    print(prompt)
    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
    print("=============== generating =================")
    generate(xfmr_weights, model_params, tokens)


if __name__ == "__main__":
    tyro.cli(main)
