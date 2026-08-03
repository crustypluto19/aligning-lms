"""
DP-BART with differential privacy noise injection at the encoder output.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Literal, Optional, Union

import mpmath
import contextlib
import torch
import torch.nn as nn
from mpmath import mp
from transformers import BartConfig, BartForConditionalGeneration

NoiseMethod = Literal["laplace", "gaussian", "analytic_gaussian"]


# ============================================================================
# Analytic Gaussian mechanism calibration (Balle & Wang, 2018)
# ============================================================================

def calibrate_analytic_gaussian(
    epsilon: float,
    delta: float,
    sensitivity: float,
    tol: float = 1e-12,
) -> float:
    """
    Calibrates the analytic Gaussian mechanism.
    Returns noise standard deviation sigma for (epsilon, delta)-DP.
    Uses mpmath for precision up to epsilon ~= 5000.
    """
    if epsilon <= 1000:
        mp.dps = 500
    elif epsilon <= 2500:
        mp.dps = 1100
    else:
        mp.dps = 2200

    def phi(t):
        return 0.5 * (1.0 + mpmath.erf(t / mpmath.sqrt(2.0)))

    def case_a(eps, s):
        return phi(mpmath.sqrt(eps * s)) - mpmath.exp(eps) * phi(-mpmath.sqrt(eps * (s + 2.0)))

    def case_b(eps, s):
        return phi(-mpmath.sqrt(eps * s)) - mpmath.exp(eps) * phi(-mpmath.sqrt(eps * (s + 2.0)))

    def doubling(predicate_stop, s_sup):
        s_inf = 0.0
        while not predicate_stop(s_sup):
            s_inf, s_sup = s_sup, 2.0 * s_sup
        return s_inf, s_sup

    def bsearch(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = (s_inf + s_sup) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = (s_inf + s_sup) / 2.0
        return s_mid

    delta_threshold = case_a(epsilon, 0.0)
    if delta == delta_threshold:
        alpha = 1.0
    else:
        if delta > delta_threshold:
            stop_doubling = lambda s: case_a(epsilon, s) >= delta
            s_to_delta = lambda s: case_a(epsilon, s)
            left_of_target = lambda s: s_to_delta(s) > delta
            s_to_alpha = lambda s: mpmath.sqrt(1.0 + s / 2.0) - mpmath.sqrt(s / 2.0)
        else:
            stop_doubling = lambda s: case_b(epsilon, s) <= delta
            s_to_delta = lambda s: case_b(epsilon, s)
            left_of_target = lambda s: s_to_delta(s) < delta
            s_to_alpha = lambda s: mpmath.sqrt(1.0 + s / 2.0) + mpmath.sqrt(s / 2.0)

        stop_bsearch = lambda s: abs(s_to_delta(s) - delta) <= tol
        s_inf, s_sup = doubling(stop_doubling, 1.0)
        s_final = bsearch(stop_bsearch, left_of_target, s_inf, s_sup)
        alpha = s_to_alpha(s_final)

    return float(alpha * sensitivity / mpmath.sqrt(2.0 * epsilon))


# ============================================================================
# DP-BART Aligned model
# ============================================================================


from transformers import BartConfig

class DPBartConfig(BartConfig):
    model_type = "dp-bart"

    def __init__(
        self,
        epsilon: float = 750.0,
        delta: float = 1e-5,
        sigma: float = 0.2,
        num_sigmas: float = 0.5,
        noise_method: str = "gaussian",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = sigma
        self.num_sigmas = num_sigmas
        self.noise_method = noise_method
        self.c_min = -sigma
        self.c_max = num_sigmas * sigma

import contextlib

class DPBartAligned(BartForConditionalGeneration):
    config_class = DPBartConfig

    def __init__(self, config: DPBartConfig):
        super().__init__(config)
        self.sync_dpo_noise = False
        self.noise_scale = self._compute_noise_scale(k=config.d_model)

        self.get_encoder().register_forward_hook(self._encoder_forward_hook)

    @contextlib.contextmanager
    def synchronized_noise(self):
        old_val = self.sync_dpo_noise
        self.sync_dpo_noise = True
        try:
            yield
        finally:
            self.sync_dpo_noise = old_val

    def _compute_noise_scale(self, k: int) -> float:
        if self.config.noise_method == "laplace":
            sensitivity = 2 * self.config.sigma * self.config.num_sigmas * k
            return sensitivity / self.config.epsilon
        if self.config.noise_method == "gaussian":
            sensitivity = 2 * self.config.sigma * self.config.num_sigmas * math.sqrt(k)
            return math.sqrt(
                (sensitivity ** 2 / self.config.epsilon ** 2) * 2 * math.log(1.25 / self.config.delta)
            )
        if self.config.noise_method == "analytic_gaussian":
            sensitivity = 2 * self.config.sigma * self.config.num_sigmas * math.sqrt(k)
            return calibrate_analytic_gaussian(self.config.epsilon, self.config.delta, sensitivity)
        raise ValueError(f"Unknown noise method: {self.config.noise_method}")

    def _inject_dp_noise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        clipped = torch.clip(hidden_states, self.config.c_min, self.config.c_max)
        batch_size, _, k = clipped.shape

        effective_bsz = batch_size // 2 if self.sync_dpo_noise else batch_size

        if self.config.noise_method == "laplace":
            dist = torch.distributions.Laplace(
                torch.zeros(effective_bsz, 1, k, device=clipped.device),
                torch.full((effective_bsz, 1, k), self.noise_scale, device=clipped.device),
            )
            noise = dist.sample()
        else:  # gaussian or analytic_gaussian
            noise = torch.normal(
                mean=0.0,
                std=self.noise_scale,
                size=(effective_bsz, 1, k),
                device=clipped.device,
            )

        if self.sync_dpo_noise:
            noise = torch.cat([noise, noise], dim=0)

        return clipped + noise

    def _encoder_forward_hook(self, module, args, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            noised = self._inject_dp_noise(hidden_states).float()
            return (noised,) + output[1:]
        else:
            hidden_states = output.last_hidden_state
            output.last_hidden_state = self._inject_dp_noise(hidden_states).float()
            return output

    @classmethod
    def from_baseline(
        cls,
        model_name: str = "facebook/bart-base",
        epsilon: float = 750.0,
        delta: float = 1e-5,
        sigma: float = 0.2,
        num_sigmas: float = 0.5,
        noise_method: str = "gaussian",
    ) -> "DPBartAligned":
        config = DPBartConfig.from_pretrained(
            model_name,
            epsilon=epsilon,
            delta=delta,
            sigma=sigma,
            num_sigmas=num_sigmas,
            noise_method=noise_method
        )
        return cls.from_pretrained(model_name, config=config)

    @classmethod
    def from_aligned_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        base_model: str = "facebook/bart-base",
        epsilon: float = 750.0,
        delta: float = 1e-5,
        sigma: float = 0.2,
        num_sigmas: float = 0.5,
        noise_method: str = "gaussian",
    ) -> "DPBartAligned":
        config = DPBartConfig.from_pretrained(
            base_model,
            epsilon=epsilon,
            delta=delta,
            sigma=sigma,
            num_sigmas=num_sigmas,
            noise_method=noise_method
        )
        model = cls.from_pretrained(base_model, config=config)
        apply_decoder_isolation(model)

        state_dict = _load_state_dict(Path(checkpoint_dir))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[DPBartAligned] Unexpected keys: {unexpected}")
        if missing:
            print(f"[DPBartAligned] Missing keys (likely tied): {missing}")
        return model

# ============================================================================
# Privatization Pipeline
# ============================================================================

class PrivatizationPipeline:
    def __init__(self, model: DPBartAligned, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    @torch.no_grad()
    def __call__(
        self,
        text: str,
        max_input_length: int = 512,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> str:
        inputs = self.tokenizer(
            text,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        num_input_tokens = inputs.input_ids.shape[-1]
        
        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens or num_input_tokens,
            **generation_kwargs,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()



# ============================================================================
# Architecture utilities
# ============================================================================

def apply_decoder_isolation(model: DPBartAligned) -> DPBartAligned:
    """
    Unties decoder embeddings and lm_head from the frozen encoder.
    BART shares embeddings between encoder, decoder, and lm_head. Freezing the 
    encoder freezes the decoder's embeddings. This deep-copies shared 
    embeddings so the decoder remains trainable.
    """
    for p in model.model.shared.parameters():
        p.requires_grad = False
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    model.model.decoder.embed_tokens = copy.deepcopy(model.model.shared)
    for p in model.model.decoder.embed_tokens.parameters():
        p.requires_grad = True

    # Retie lm_head to new decoder embeddings.
    model.lm_head = nn.Linear(
        model.config.d_model, model.config.vocab_size, bias=False
    )
    model.lm_head.weight = model.model.decoder.embed_tokens.weight
    model.lm_head.to(model.device)

    return model


def freeze_reference_model(ref_model: DPBartAligned) -> DPBartAligned:
    """
    Prepares a DPO reference model by mirroring policy architecture and freezing parameters.
    TRL's DPOTrainer requires matching parameter structures.
    """
    ref_model.model.decoder.embed_tokens = copy.deepcopy(ref_model.model.shared)
    ref_model.lm_head = nn.Linear(
        ref_model.config.d_model, ref_model.config.vocab_size, bias=False
    )
    ref_model.lm_head.weight = ref_model.model.decoder.embed_tokens.weight
    ref_model.lm_head.to(ref_model.device)

    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


def load_dp_bart(
    model_name: str = "facebook/bart-base",
    epsilon: float = 750.0,
    delta: float = 1e-5,
    sigma: float = 0.2,
    num_sigmas: float = 0.5,
    noise_method: NoiseMethod = "gaussian",
) -> DPBartAligned:
    """Module-level alias for `DPBartAligned.from_baseline`."""
    return DPBartAligned.from_baseline(
        model_name=model_name,
        epsilon=epsilon,
        delta=delta,
        sigma=sigma,
        num_sigmas=num_sigmas,
        noise_method=noise_method,
    )


# ============================================================================
# State dict loading (safetensors-aware)
# ============================================================================

def _load_state_dict(checkpoint_dir: Path) -> dict:
    """Load a state_dict from a checkpoint directory."""
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path), device="cpu")
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
    )
