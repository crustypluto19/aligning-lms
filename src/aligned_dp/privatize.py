"""
Generate privatized text datasets using a DP-BART model.
Supports baseline, aligned, and PR+ configurations.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path
from typing import Callable, Optional, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import BartTokenizer

from .dp_bart_aligned import DPBartAligned, PrivatizationPipeline
from .dp_bart_pr_plus import DPBartPRPlus, available_pr_plus_epsilons



DEFAULT_INPUT_DIR = Path("./data/sampled_eval_datasets")
DEFAULT_OUTPUT_DIR = Path("./data/eval_datasets/privatized")
DEFAULT_BASE_MODEL = "facebook/bart-base"
DEFAULT_SEED = 42

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



class UnsupportedEpsilonError(Exception):
    pass

class CSVDatasetAdapter:
    """CSV adapter for dataset text extraction and resume state."""
    TEXT_COLUMN_PER_DATASET = {
        "yelp": "review",
        "trustpilot": "text",
        "reddit-mental-health": "text",
    }
    
    def __init__(self, dataset: str, input_path: Path, output_path: Path, save_every: int = 20):
        self.dataset = dataset
        self.input_path = input_path
        self.output_path = output_path
        self.save_every = save_every
        
        if dataset not in self.TEXT_COLUMN_PER_DATASET:
            raise ValueError(f"Unknown dataset '{dataset}'. Add its text column to TEXT_COLUMN_PER_DATASET.")
        self.text_col = self.TEXT_COLUMN_PER_DATASET[dataset]
        
        if output_path.exists():
            self.df = pd.read_csv(output_path)
            completed = self.df["privatized_text"].notna() & (self.df["privatized_text"].astype(str) != "")
            self.start_idx = int(completed.sum())
        else:
            self.df = pd.read_csv(input_path).dropna(subset=[self.text_col])
            self.df = self.df.rename(columns={self.text_col: "original_text"})
            self.df["privatized_text"] = ""
            self.start_idx = 0

    def unprocessed_items(self) -> Iterable[tuple[int, str]]:
        if self.start_idx >= len(self.df):
            print(f"  [skip] {self.output_path.name} is already complete.")
            return []
            
        if self.start_idx > 0:
            print(f"  [resume] {self.output_path.name} from row {self.start_idx}/{len(self.df)}")
            

        pbar = tqdm(range(self.start_idx, len(self.df)), desc=f"  {self.dataset}", leave=False)
        for i in pbar:
            text = str(self.df.iloc[i]["original_text"])
            yield i, text
            
    def save_result(self, index: int, privatized_text: str) -> None:
        self.df.at[index, "privatized_text"] = privatized_text
        if (index + 1) % self.save_every == 0:
            self.df.to_csv(self.output_path, index=False)
            
    def finalize(self) -> None:
        self.df.to_csv(self.output_path, index=False)


class DatasetPrivatizer:
    """Coordinates DatasetAdapter and PrivatizationPipeline."""
    def __init__(self, pipeline: Callable[[str], str]):
        self.pipeline = pipeline
        
    def privatize(self, adapter: CSVDatasetAdapter) -> None:
        for idx, text in adapter.unprocessed_items():
            try:
                result = self.pipeline(text)
            except Exception as e:
                print(f"\n    [error row {idx}] {type(e).__name__}: {e}", file=sys.stderr)
                result = ""
            adapter.save_result(idx, result)
        adapter.finalize()



ModelFactory = Callable[[float], Callable[[str], str]]  # returns a pipeline


def make_baseline_factory(size: str) -> ModelFactory:
    model_name = f"facebook/bart-{size}"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    device = get_device()

    def build(epsilon: float):
        model = DPBartAligned.from_baseline(model_name, epsilon=epsilon).to(device)
        model.eval()
        return PrivatizationPipeline(model, tokenizer)

    return build


def make_aligned_factory(checkpoint_dir: Path, base_model: str) -> ModelFactory:
    tokenizer = BartTokenizer.from_pretrained(base_model)
    device = get_device()

    def build(epsilon: float):
        model = DPBartAligned.from_aligned_checkpoint(
            checkpoint_dir, base_model=base_model, epsilon=epsilon,
        ).to(device)
        model.eval()
        return PrivatizationPipeline(model, tokenizer)

    return build


def make_pr_plus_factory(base_model: str) -> ModelFactory:
    tokenizer = BartTokenizer.from_pretrained(base_model)
    device = get_device()
    valid_eps = available_pr_plus_epsilons()

    def build(epsilon: float):
        if epsilon not in valid_eps:
            raise UnsupportedEpsilonError(f"PR+ does not support epsilon {epsilon}. Valid: {valid_eps}")
        model = DPBartPRPlus.from_huggingface_hub(
            epsilon=epsilon, base_model=base_model, device=device,
        )
        model.eval()
        return lambda text: model.privatize(text, tokenizer)

    return build



def run(
    args: argparse.Namespace,
    factory: ModelFactory,
    config_tag: str,
) -> None:
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode      : {args.command}")
    print(f"Config tag: {config_tag}")
    print(f"Datasets  : {', '.join(args.datasets)}")
    print(f"Epsilons  : {', '.join(str(e) for e in args.epsilons)}")
    print(f"Input dir : {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device    : {get_device()}")
    print(f"Seed      : {args.seed}")
    print()

    for eps in args.epsilons:
        print(f"=== epsilon = {eps} ===")
        try:
            pipeline = factory(eps)
        except UnsupportedEpsilonError as e:
            print(f"[skip] {e}")
            print()
            continue

        privatizer = DatasetPrivatizer(pipeline)

        for dataset in args.datasets:
            input_path = args.input_dir / f"{dataset}.csv"
            if not input_path.exists():
                print(f"  [skip] input not found: {input_path}")
                continue

            output_path = args.output_dir / f"{config_tag}_eps{int(eps)}_{dataset}.csv"
            adapter = CSVDatasetAdapter(dataset, input_path, output_path)
            privatizer.privatize(adapter)

        del pipeline
        clear_vram()
        print()




def _add_common_args(parser: argparse.ArgumentParser) -> None:
    # Use the static dict keys for help text
    known_datasets = sorted(CSVDatasetAdapter.TEXT_COLUMN_PER_DATASET)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help=f"Dataset names. Known: {known_datasets}")
    parser.add_argument("--epsilons", nargs="+", type=float, required=True,
                        help="DP epsilon values to generate.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing <dataset>.csv inputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write privatized CSVs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Process RNG seed for reproducibility.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: baseline
    baseline = sub.add_parser("baseline", help="Untrained DP-BART-CLV.")
    _add_common_args(baseline)
    baseline.add_argument("--size", choices=["base", "large"], default="base",
                          help="facebook/bart-{size}.")

    # Subcommand: aligned
    aligned = sub.add_parser("aligned", help="DPO/VPO-trained checkpoint.")
    _add_common_args(aligned)
    aligned.add_argument("--checkpoint", type=Path, required=True,
                         help="Path to the checkpoint directory saved by train.py.")
    aligned.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                         help="Base BART model used during training.")
    aligned.add_argument("--config-name", default=None,
                         help="Tag used in output filenames. Defaults to the checkpoint directory name.")

    pr_plus = sub.add_parser("pr-plus", help="DP-BART-PR+ baseline from HF Hub.")
    _add_common_args(pr_plus)
    pr_plus.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                         help="Base BART tokenizer/config to use.")

    return parser.parse_args()




def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.command == "baseline":
        factory = make_baseline_factory(args.size)
        config_tag = f"dpbart_{args.size}"

    elif args.command == "aligned":
        if not args.checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
        factory = make_aligned_factory(args.checkpoint, args.base_model)
        config_tag = args.config_name or args.checkpoint.name

    elif args.command == "pr-plus":
        factory = make_pr_plus_factory(args.base_model)
        config_tag = "dpbart_prplus"

    else:
        raise RuntimeError(f"Unknown command: {args.command}")

    run(args, factory, config_tag)

if __name__ == "__main__":
    main()
