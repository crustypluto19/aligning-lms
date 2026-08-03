"""
DP-BART-PR+ baseline (Igamberdiev & Habernal, 2023) for inference only.
Requires padding inputs to MAX_SEQ_LEN for valid noise calibration.
Pruned channels are zero-filled before decoder runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .dp_utils import (
    calibrateAnalyticGaussianMechanism_precision,
    load_neurons_for_pruning,
    non_intersection,
)


# Pretrained PR+ checkpoints on HuggingFace Hub.
_PR_PLUS_REPOS: dict[float, dict[str, str]] = {
    250.0: {
        "repo": "TrustHLT/DP-BART-PR-Plus_Epsilon250",
        "weights": "dp_bart_pr_plus_ep250.pt",
        "pruning_indices": "k_prune_neurons_iter5.pt",
    },
    750.0: {
        "repo": "TrustHLT/DP-BART-PR-Plus_Epsilon750",
        "weights": "dp_bart_pr_plus_ep750.pt",
        "pruning_indices": "k_prune_neurons_iter5.pt",
    },
}


def available_pr_plus_epsilons() -> list[float]:
    """Return the epsilon values for which pretrained PR+ weights exist."""
    return list(_PR_PLUS_REPOS.keys())


class DPBartPRPlus(BartForConditionalGeneration):
    """DP-BART-PR+ for privatized text generation."""

    MAX_SEQ_LEN: int = 512

    # Fixed by architecture to maintain pretrained weights validity.
    sigma: float = 0.1
    mean: float = 0.0
    num_sigmas: float = 1.0

    def __init__(
        self,
        config: BartConfig,
        epsilon: float,
        pruning_indices_path: Union[str, Path],
        delta: float = 1e-5,
    ):
        super().__init__(config)
        self.epsilon = epsilon
        self.delta = delta
        self.hidden_dim = config.d_model  # 768 for bart-base

        # Registered non-persistently to allow .to(device) without saving to state_dict.
        indices = load_neurons_for_pruning(in_path=str(pruning_indices_path))
        self.register_buffer("pruning_indices", indices, persistent=False)

        # Sensitivity counts non-pruned channels across padded sequence length. Fixed by architecture.
        self.k = self.MAX_SEQ_LEN * (self.hidden_dim - self.pruning_indices.shape[0])
        self.sensitivity = 2 * self.sigma * self.num_sigmas * np.sqrt(self.k)
        self.noise_scale = calibrateAnalyticGaussianMechanism_precision(
            epsilon, delta, self.sensitivity,
        )

    # DP configuration
    def update_epsilon(self, new_epsilon: float) -> None:
        """Update epsilon and recompute the noise scale."""
        self.epsilon = new_epsilon
        self.noise_scale = calibrateAnalyticGaussianMechanism_precision(
            new_epsilon, self.delta, self.sensitivity,
        )

    # Pruning helpers
    def _reduce_pruned(self, hidden: torch.Tensor):
        """Slice out only the non-pruned channels along the last dim."""
        all_idx = torch.arange(self.hidden_dim, device=hidden.device)
        non_pruned_idx = non_intersection(self.pruning_indices, all_idx)
        return hidden[:, :, non_pruned_idx], non_pruned_idx

    def _rebuild_pruned(
        self,
        non_pruned_hidden: torch.Tensor,
        non_pruned_idx: torch.Tensor,
        batch_dim: int,
    ) -> torch.Tensor:
        """Rebuild a full (batch, seq, d_model) tensor with pruned positions zeroed."""
        full = torch.zeros(
            (batch_dim, self.MAX_SEQ_LEN, self.hidden_dim),
            device=non_pruned_hidden.device,
        )
        full[:, :, non_pruned_idx] = non_pruned_hidden
        return full

    # Privatized generation
    @torch.no_grad()
    def privatize(
        self,
        text: str,
        tokenizer,
        max_new_tokens: int = 150,
        num_beams: int = 4,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 2,
        **generation_kwargs,
    ) -> str:
        """Run the full DP-BART-PR+ pipeline on a single input string."""
        # Padding to MAX_SEQ_LEN required to maintain privacy guarantee.
        inputs = tokenizer(
            text,
            max_length=self.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        encoder_outputs = self.model.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        hidden = encoder_outputs.last_hidden_state
        batch_dim = hidden.shape[0]

        # Zero pruned channels (they carry no signal under PR+).
        hidden[:, :, self.pruning_indices] = 0

        non_pruned, non_pruned_idx = self._reduce_pruned(hidden)
        smaller_dim = non_pruned.shape[-1]
        flat = non_pruned.view(batch_dim, -1)

        left = self.mean - self.sigma * self.num_sigmas
        right = self.mean + self.sigma * self.num_sigmas
        flat_clipped = torch.clamp(flat, left, right)

        noise = torch.normal(
            mean=0.0,
            std=self.noise_scale,
            size=flat_clipped.shape,
            device=flat_clipped.device,
        )
        flat_noised = flat_clipped + noise

        noised = flat_noised.view(batch_dim, self.MAX_SEQ_LEN, smaller_dim)
        full_hidden = self._rebuild_pruned(noised, non_pruned_idx, batch_dim)

        output_ids = self.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=full_hidden),
            attention_mask=inputs.attention_mask,  # masks out padding for the decoder
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **generation_kwargs,
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # HuggingFace Hub loader
    @classmethod
    def from_huggingface_hub(
        cls,
        epsilon: float,
        base_model: str = "facebook/bart-base",
        device: Optional[torch.device] = None,
    ) -> "DPBartPRPlus":
        """
        Downloads and loads pretrained PR+ weights from HuggingFace Hub.
        """
        if epsilon not in _PR_PLUS_REPOS:
            raise ValueError(
                f"DP-BART-PR+ pretrained weights are only available for "
                f"epsilon in {sorted(_PR_PLUS_REPOS)}; got {epsilon}."
            )
        info = _PR_PLUS_REPOS[epsilon]

        pruning_path = hf_hub_download(repo_id=info["repo"], filename=info["pruning_indices"])
        weights_path = hf_hub_download(repo_id=info["repo"], filename=info["weights"])

        config = BartConfig.from_pretrained(base_model)
        model = cls(config=config, epsilon=epsilon, pruning_indices_path=pruning_path)

        # weights_only=False required for legacy checkpoint format.
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)

        if device is not None:
            model = model.to(device)
        return model
