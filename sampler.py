from typing import *

import mlx
import mlx.core as mx
import mlx.nn

from utils import COLORS


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


class SamplerConfig:
    """
    Encapsulation of all available sampler hyperparameters.

    This should be a good starting point for baselining experiments.
    """
    temp: float = 0.666
    top_p: float = 0.95
    top_k: int = 27
    min_p: float = 0.03  # Turn this down to 0.01 to reduce the shoggoth

    low_ent_thresh: float = 7.0
    low_vent_thresh: float = 7.0
    med_ent_thresh: float = 10.0
    med_vent_thresh: float = 10.0
    high_ent_thresh: float = 13.0
    high_vent_thresh: float = 13.0
    
    # Attention Entropy Thresholds
    low_attention_entropy_threshold: float = 11.915
    medium_attention_entropy_threshold: float = 11.921
    high_attention_entropy_threshold: float = 11.926

    # Attention Varentropy Thresholds
    low_attention_varentropy_threshold: float = 0.001
    medium_attention_varentropy_threshold: float = 0.0045
    high_attention_varentropy_threshold: float = 0.009

    # Agreement Thresholds
    low_agreement_threshold: float = 2e-06
    medium_agreement_threshold: float = 4e-06
    high_agreement_threshold: float = 5e-06

    # Interaction Strength Thresholds
    low_interaction_strength_threshold: float = 0.2
    medium_interaction_strength_threshold: float = 0.247
    high_interaction_strength_threshold: float = 0.264

    # TODO this is a bit of a nasty mess, but also makes all the hyperparameters visible
    helv_attn_ent_offset: float = 1.3
    helv_attn_ent_coef: float = 0.2

    lehv_interaction_strength_offset: float = 1.2
    lehv_interaction_strength_coef: float = 0.3

    hehv_attn_ent_coef: float = 0.2
    hehv_attn_vent_offset: float = 2.0
    hehv_attn_vent_coef: float = 0.5

    # TODO not convinced this should
    n_adaptive_samples: int = 5

    # Adaptive sampling parameters
    ada_temp_logits: float = 0.3
    ada_temp_attn: float = 0.2
    ada_temp_agree: float = 0.2
    ada_top_p: float = 0.1
    ada_top_k_int: float = 0.3
    ada_top_k_agree: float = 0.2
    ada_min_p: float = 0.5
    ada_score_logits_ent: float = 0.1
    ada_score_attn_ent: float = 0.2
    ada_score_logits_vent: float = 0.3
    ada_score_attn_vent: float = 0.4
    ada_score_agree: float = 0.5
    ada_score_int: float = 0.6



# pure function, compile
@mx.compile
def calculate_varentropy_logsoftmax(logits: mx.array, axis: int = -1) -> Tuple[mx.array, mx.array]:
    """
    Entropy and varentropy from logits using log softmax function.
    """    
    log_probs = mlx.nn.log_softmax(logits, axis=axis)
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=axis) / LN_2
    varentropy = mx.sum(probs * (log_probs / LN_2 + mx.expand_dims(entropy, -1))**2, axis=axis)
    return entropy, varentropy


def multinominal_sample_one(probs_sort: mx.array, key: mx.array) -> mx.array:
    """
    Samples one token from a multinomial distribution with sorted probabilities.
    """
    # mlx does not have the exponential random distribution
    # but we can model it using the uniform distribution like below
    # taking 1 - u to move the domain of log to (0, 1] instead of [0, 1)
    u = mx.random.uniform(key=key, shape=probs_sort.shape)
    q = -mx.log1p(mx.negative(u))
    return mx.argmax(probs_sort / q, axis=-1, keepdims=True).astype(mx.int32)


def flip(x: mx.array, axis: int = -1):
    """
    Reverse the order of elements along a given axis
    """
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(None, None, -1)
    return x[tuple(slices)]


def calculate_metrics(logits: mx.array, attention_scores: mx.array) -> Dict[str, mx.array]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = mlx.nn.softmax(attention_scores, axis=-1)
    attn_entropy = -mx.sum(attention_probs * mx.log2(mx.clip(attention_probs, 1e-7, 1.0)), axis=-1)
    attn_varentropy = mx.var(attn_entropy, axis=1)

    mean_attention = mx.mean(attention_probs, axis=1)
    agreement = mx.mean(mx.abs(attention_probs - mx.expand_dims(mean_attention, 1)), axis=(1, 2))

    interaction_strength = mx.mean(mx.abs(attention_scores), axis=(1, 2, 3))

    return dict(
        logits_entropy=mx.mean(entropy),
        logits_varentropy=mx.mean(varentropy),
        attn_entropy=mx.mean(attn_entropy),
        attn_varentropy=mx.mean(attn_varentropy),
        agreement=mx.mean(agreement),
        interaction_strength=interaction_strength
    )


def _in1d(element: mx.array, test_elements: mx.array, invert: bool = False) -> mx.array:
    arr1, arr2 = element.flatten(), test_elements.flatten()
    if arr1.size == 0 or arr2.size == 0:
        return mx.ones(arr1.shape, dtype=mx.bool_) if invert else mx.zeros(arr1.shape, dtype=mx.bool_)
    if invert:
        return (mx.expand_dims(arr1, -1) != mx.expand_dims(arr2, 0)).all(-1)
    return (mx.expand_dims(arr1, -1) == mx.expand_dims(arr2, 0)).any(-1)


def isin(element: mx.array, test_elements: mx.array, invert: bool = False) -> mx.array:
    """
    hacky isin function to mimic `jax.numpy.isin`
    """
    ele = mx.array(element) if not isinstance(element, mx.array) else element
    tele = mx.array(test_elements) if not isinstance(test_elements, mx.array) else test_elements
    result = _in1d(ele, tele, invert=invert)
    return result.reshape(element.shape)


def _sample(
        logits: mx.array, *, temperature: Union[float, mx.array], top_p: Union[float, mx.array], 
        top_k: Union[int, mx.array], min_p: Union[float, mx.array], key: mx.array = None
    ) -> mx.array:
    if key is None:
        key = mx.random.key(1337)

    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = mlx.nn.softmax(logit / temperature, axis=-1)
    
    # apply min_p sampling
    if min_p > 0.0:
        p_max = mx.max(probs, axis=-1, keepdims=True)
        indices_to_remove = probs < (min_p * p_max)
        replacement = mx.ones_like(logit) * float('-inf')
        logit = mx.where(indices_to_remove, replacement, logit)

    # apply top-k sampling
    _indices = mx.argsort(-probs, axis=-1)
    top_k_indices = mx.take(_indices, mx.arange(top_k), axis=-1)
    top_k_probs = mx.take_along_axis(probs, top_k_indices, axis=-1)
    probs_sort = flip(top_k_probs, axis=-1)
    probs_idx = flip(top_k_indices, axis=-1)
    probs_sum = mx.cumsum(probs_sort, axis=-1)

    # apply top_p sampling
    mask = mx.where(probs_sum - probs_sort > top_p, 1.0, 0.0)
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / mx.sum(probs_sort, axis=-1, keepdims=True)
    next_token = multinominal_sample_one(probs_sort, key)
    next_token_g = mx.take_along_axis(probs_idx, next_token.reshape(bsz, 1), axis=-1)
    return next_token_g.astype(mx.int32)


# our hero
def sample(
        gen_tokens: mx.array, logits: mx.array, attention_scores: mx.array, cfg: SamplerConfig,
        clarifying_question_token: int = 2564, key: mx.array = None
    ) -> Tuple[mx.array, str, dict]:
    if key is None:
        key = mx.random.key(1337)
    
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # low entropy, low varentropy = "flowing with unspoken intent"
    # the model is very certain, choose the most likely token
    if (ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh and 
        attn_ent < cfg.low_attention_entropy_threshold and
        attn_vent < cfg.low_attention_varentropy_threshold and
        agreement < cfg.low_agreement_threshold and
        interaction_strength < cfg.low_interaction_strength_threshold):
        return mx.argmax(logits[:, -1], axis=-1, keepdims=True).astype(mx.int32), COLORS["lelv"], metrics
    
    # high entropy, low varentropy = "treading carefully, asking clarifying questions"
    # the model is uncertain but consistently so, leading to careful sampling or 
    # asking clarifying questions.
    elif (ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength < cfg.low_interaction_strength_threshold):
        if not isin(gen_tokens[:, -1], clarifying_question_token).any():
            return mx.array([[clarifying_question_token]]), COLORS["hehv"], metrics
        else:
            # if we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * attn_ent
            return _sample(logits, temperature=min(1.5, cfg.temp * temp_adj), top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p, key=key), COLORS["helv"], metrics
        
    # low entropy, high varentropy: "exploring forks in the path"
    elif (ent < cfg.low_ent_thresh and vent > cfg.high_vent_thresh and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength > cfg.low_interaction_strength_threshold):
        print("(lehv)", flush = True, end = "")
        temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(logits, temperature=min(1.5, cfg.temp * temp_adj), top_p=cfg.top_p, top_k=top_k_adj, min_p=cfg.min_p, key=key), COLORS["lehv"], metrics

    # high entropy, high varentropy: "resampling in the mist"
    elif (ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh and 
          attn_ent > cfg.high_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement > cfg.high_agreement_threshold and
          interaction_strength > cfg.high_interaction_strength_threshold):
        print("(hehv)", flush = True, end = "")
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(logits, temperature=max(2.0, cfg.temp * temp_adj), top_p=top_p_adj, top_k=cfg.top_k, min_p=cfg.min_p, key=key), COLORS["hehv"], metrics
    
    # middle ground: use adaptive sampling
    else:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = cfg.temp * (1 + cfg.ada_temp_logits * ent + cfg.ada_temp_attn * attn_ent - cfg.ada_temp_agree * metrics["agreement"])
        top_p = mx.clip(cfg.top_p * (1 + cfg.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = mx.clip(
            mx.round(cfg.top_k * (1 + cfg.ada_top_k_int * metrics["interaction_strength"].item() - cfg.ada_top_k_agree * metrics["agreement"].item())),
            a_min=1,
            a_max=100
        ).astype(mx.uint32).item()
        min_p = mx.clip(cfg.min_p * (1 - cfg.ada_min_p * vent), 0.01, 0.5)

        keys = mx.random.split(key, cfg.n_adaptive_samples)

        # basically, sample n(5) number of times
        # choose the best from it
        samples = []
        for sample_key in keys:
            sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, key=sample_key)
            samples.append(sample)

        def score_sample(sample: mx.array):
            bsz, seqlen = sample.shape
            vbsz = logits.shape[-1]
            one_hot = mx.zeros((bsz, seqlen, vbsz))
            one_hot[mx.arange(bsz)[:, None], mx.arange(seqlen)[None, :], sample] = 1
            log_prob = mx.sum(mlx.nn.log_softmax(logits) * one_hot)
            confidence_score = (
                (1 - ent / cfg.high_ent_thresh) * cfg.ada_score_logits_ent +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.ada_score_attn_ent +
                (1 - vent / cfg.high_vent_thresh) * cfg.ada_score_logits_vent +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.ada_score_attn_vent +
                (agreement / cfg.high_agreement_threshold) * cfg.ada_score_agree +
                (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.ada_score_int
            )
            return log_prob + confidence_score
        
        sample_scores = [score_sample(sample) for sample in samples]
        best_sample_idx = mx.argmax(mx.array(sample_scores)).item()
        return samples[best_sample_idx], COLORS["ada"], metrics
