"""
Preference-aligned differentially private text rewriting.
Extends DP-BART-CLV with a decoder alignment phase (DPO/VPO).
Scoring stack (`metric_acceptability`) not imported by default to avoid heavy dependencies.
"""

from .dp_bart_aligned import (
    DPBartAligned,
    PrivatizationPipeline,
    apply_decoder_isolation,
    freeze_reference_model,
    load_dp_bart,
)
from .dp_bart_pr_plus import DPBartPRPlus, available_pr_plus_epsilons
from .vpo_trainer import VPOTrainer

__version__ = "0.1.0"

__all__ = [
    "DPBartAligned",
    "PrivatizationPipeline",
    "apply_decoder_isolation",
    "freeze_reference_model",
    "load_dp_bart",
    "DPBartPRPlus",
    "available_pr_plus_epsilons",
    "VPOTrainer",
]
