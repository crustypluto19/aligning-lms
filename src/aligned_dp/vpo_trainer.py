"""
DPOTrainer subclass with soft preference labels and synchronized noise injection.
Implements the VPO objective: L = p * BCE(chosen) + (1 - p) * BCE(rejected).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from trl import DPOTrainer


class VPOTrainer(DPOTrainer):
    """DPOTrainer with soft VPO labels and synchronized chosen/rejected noise."""

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        # Synchronized noise between chosen/rejected.
        with model.synchronized_noise():
            policy_out = self.concatenated_forward(model, batch)

        policy_chosen_logps = policy_out["chosen_logps"]
        policy_rejected_logps = policy_out["rejected_logps"]

        # Reference pass with noise synchronized within model.
        with torch.no_grad():
            with self.ref_model.synchronized_noise():
                ref_out = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)

            ref_chosen_logps = ref_out["chosen_logps"]
            ref_rejected_logps = ref_out["rejected_logps"]

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        reward_gap = self.beta * (pi_logratios - ref_logratios)

        # Soft VPO objective. Falls back to standard DPO BCE if no p_target.
        if "p_target" in batch:
            p = batch["p_target"].to(reward_gap.device)
            loss_chosen = -F.logsigmoid(reward_gap)
            loss_rejected = -F.logsigmoid(-reward_gap)
            losses = p * loss_chosen + (1 - p) * loss_rejected
        else:
            losses = -F.logsigmoid(reward_gap)

        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean().cpu(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().cpu(),
            f"{prefix}rewards/accuracies": reward_accuracies.mean().cpu(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean().cpu(),
            f"{prefix}logps/chosen": policy_chosen_logps.detach().mean().cpu(),
            f"{prefix}logps/rejected": policy_rejected_logps.detach().mean().cpu(),
        }
        if "chosen_logits" in policy_out:
            metrics[f"{prefix}logits/chosen"] = policy_out["chosen_logits"].detach().mean().cpu()
        if "rejected_logits" in policy_out:
            metrics[f"{prefix}logits/rejected"] = policy_out["rejected_logits"].detach().mean().cpu()

        return losses.mean(), metrics
