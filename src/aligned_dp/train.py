"""
Train DPBartAligned with DPO or VPO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from datasets import load_dataset
from transformers import BartTokenizer
from trl import DPOConfig, DPOTrainer

from .dp_bart_aligned import (
    DPBartAligned,
    apply_decoder_isolation,
    freeze_reference_model,
    load_dp_bart,
)
from .vpo_trainer import VPOTrainer


DEFAULT_MODEL = "facebook/bart-base"
DEFAULT_PROJECT = "aligned-dp-thesis"




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPBartAligned with DPO or VPO.")
    # Dataset and DP.
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to a JSONL preference dataset.")
    parser.add_argument("--epsilon", type=float, default=750.0,
                        help="DP epsilon. Must match the value used at evaluation.")
    parser.add_argument("--delta", type=float, default=1e-5, help="DP delta.")
    # Optimization.
    parser.add_argument("--beta", type=float, default=0.01,
                        help="DPO temperature; lower keeps the policy closer to the reference.")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    # Objective.
    parser.add_argument("--use-vpo", action="store_true",
                        help="Use the soft VPO objective with synchronized noise. "
                             "Requires a 'p_target' column in the dataset.")
    # Bookkeeping.
    parser.add_argument("--config-name", type=str, default=None,
                        help="Override the auto-derived config name in run naming.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--output-root", type=Path, default=Path("./models"))
    return parser.parse_args()


def _fmt(value: float) -> str:
    return str(value).replace(".", "p")


def derive_run_name(args: argparse.Namespace) -> str:
    if args.config_name:
        variant = args.config_name
    elif args.use_vpo:
        variant = "config-1.2-human-vpo"
    else:
        variant = "config-1.1-human-dpo"

    return (
        f"{variant}"
        f"-{Path(args.dataset).stem}"
        f"-beta-{_fmt(args.beta)}"
        f"-lr-{_fmt(args.learning_rate)}"
        f"-epsilon-{_fmt(args.epsilon)}"
        f"-epochs-{args.epochs}"
    )




def main() -> None:
    args = parse_args()
    run_name = derive_run_name(args)
    output_dir = args.output_root / f"dp-bart-aligned-{run_name}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        tags=["frozen-encoder", "untied-embeddings", "vpo" if args.use_vpo else "dpo"],
        config=vars(args),
    )

    tokenizer = BartTokenizer.from_pretrained(args.model)

    # Policy model: vanilla BART -> DP params -> decoder isolation.
    print(f"Loading policy model from {args.model} (epsilon={args.epsilon})")
    policy = load_dp_bart(args.model, epsilon=args.epsilon, delta=args.delta)
    apply_decoder_isolation(policy)

    # Reference model: same architecture, fully frozen.
    print("Loading reference model")
    reference = load_dp_bart(args.model, epsilon=args.epsilon, delta=args.delta)
    freeze_reference_model(reference)

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Trainable parameters (decoder + untied embeddings): {trainable:,}")


    print(f"Loading dataset from {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset, split="train")


    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        weight_decay=0.05,
        logging_steps=10,
        save_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        beta=args.beta,
        max_length=args.max_length,
    )

    trainer_cls = VPOTrainer if args.use_vpo else DPOTrainer

    trainer = trainer_cls(
        model=policy,
        ref_model=reference,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting {'VPO' if args.use_vpo else 'DPO'} training")
    trainer.train()

    print(f"Saving aligned decoder to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    wandb.finish()


if __name__ == "__main__":
    main()
