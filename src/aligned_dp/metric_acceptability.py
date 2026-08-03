"""Metric Acceptability: a composite quality score for privatized text rewrites."""

from __future__ import annotations

import gc
from enum import Enum
from typing import Optional, Sequence

import evaluate
import numpy as np
import pandas as pd
import textstat
import torch
from transformers import pipeline



class MetricDirection(Enum):
    HIGHER_IS_BETTER = 1
    LOWER_IS_BETTER = -1
    CLOSER_TO_BASELINE = 0


class MetricAdapter:
    """Interface for a single evaluation metric."""
    name: str
    direction: MetricDirection
    needs_reference: bool

    def load(self, device: str) -> None:
        """Load the underlying models into the target device."""
        pass

    def compute(
        self,
        predictions: Sequence[str],
        references: Optional[Sequence[str]] = None,
    ) -> list[float]:
        """Compute the raw metric score for a batch of predictions."""
        raise NotImplementedError

    def unload(self) -> None:
        """Evict the underlying models from memory."""
        pass


class Evaluator:
    """Engine for running metrics and managing VRAM."""
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def evaluate(
        self,
        metrics: list[MetricAdapter],
        predictions: Sequence[str],
        references: Optional[Sequence[str]] = None,
    ) -> dict[str, list[float]]:
        """Run all metrics sequentially and return scores by metric name."""
        results = {}
        for metric in metrics:
            metric.load(self.device)
            refs = references if metric.needs_reference else None
            results[metric.name] = metric.compute(predictions, refs)
            metric.unload()
            self._purge()
        return results

    def _purge(self) -> None:
        """Force GPU memory eviction."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ScoreAggregator:
    """Normalizes scores and groups them into facets."""
    def __init__(
        self,
        metrics: list[MetricAdapter],
        facets: dict[str, list[str]],
        clip: float = 3.0,
    ):
        self.metrics = metrics
        self.facets = facets
        self.clip = clip

    def aggregate(
        self,
        df: pd.DataFrame,
        baseline_stats: pd.DataFrame,
        group_col: str = "dataset",
    ) -> pd.DataFrame:
        out = df.merge(baseline_stats, on=group_col, how="left")

        for metric in self.metrics:
            raw_col = f"raw_{metric.name}"
            mu_col = f"mu_{metric.name}"
            sig_col = f"sig_{metric.name}"
            z_col = f"Z_{metric.name}"

            if metric.direction == MetricDirection.HIGHER_IS_BETTER:
                out[z_col] = self._zclip(out[raw_col], out[mu_col], out[sig_col], sign=1.0)
            elif metric.direction == MetricDirection.LOWER_IS_BETTER:
                out[z_col] = self._zclip(out[raw_col], out[mu_col], out[sig_col], sign=-1.0)
            elif metric.direction == MetricDirection.CLOSER_TO_BASELINE:
                out[z_col] = self._absclip(out[raw_col], out[mu_col], out[sig_col])

        facet_cols = []
        for facet_name, metric_names in self.facets.items():
            facet_col = f"Score_{facet_name}"
            facet_cols.append(facet_col)
            z_cols = [f"Z_{m}" for m in metric_names]
            out[facet_col] = out[z_cols].mean(axis=1)

        out["Metric_Acceptability"] = out[facet_cols].mean(axis=1)
        return out

    def _zclip(self, raw: pd.Series, mu: pd.Series, sig: pd.Series, sign: float = 1.0) -> pd.Series:
        return np.clip(sign * (raw - mu) / sig, -self.clip, self.clip)

    def _absclip(self, raw: pd.Series, mu: pd.Series, sig: pd.Series) -> pd.Series:
        return np.clip(-np.abs((raw - mu) / sig), -self.clip, self.clip)


def compute_per_dataset_stats(
    df: pd.DataFrame,
    metric_cols: dict[str, str],
    group_col: str = "dataset",
) -> pd.DataFrame:
    """Compute per-group mean and standard deviation for metrics."""
    agg = {}
    for col, suffix in metric_cols.items():
        agg[f"mu_{suffix}"] = (col, "mean")
        agg[f"sig_{suffix}"] = (col, "std")

    stats = df.groupby(group_col).agg(**agg).reset_index()
    for col in stats.columns:
        if col.startswith("sig_"):
            stats[col] = stats[col].replace(0, 1e-9)
    return stats


def sanitize(texts: Sequence[str]) -> tuple[list[str], set[int]]:
    valid_idx = {i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()}
    sanitized = [t if i in valid_idx else "." for i, t in enumerate(texts)]
    return sanitized, valid_idx




class BERTScoreAdapter(MetricAdapter):
    name = "bertscore"
    direction = MetricDirection.HIGHER_IS_BETTER
    needs_reference = True

    def __init__(self, model_type="roberta-large", batch_size=4):
        self.model_type = model_type
        self.batch_size = batch_size
        self.metric = None
        self.device = None

    def load(self, device: str) -> None:
        self.device = device
        self.metric = evaluate.load("bertscore")

    def compute(self, predictions: Sequence[str], references: Optional[Sequence[str]] = None) -> list[float]:
        sanitized, valid_idx = sanitize(predictions)
        result = self.metric.compute(
            predictions=sanitized,
            references=list(references),
            lang="en",
            model_type=self.model_type,
            device=self.device,
            batch_size=self.batch_size,
        )
        return [v if i in valid_idx else 0.0 for i, v in enumerate(result["f1"])]

    def unload(self) -> None:
        del self.metric
        self.metric = None


class BLEURTAdapter(MetricAdapter):
    name = "bleurt"
    direction = MetricDirection.HIGHER_IS_BETTER
    needs_reference = True

    def __init__(self, checkpoint="BLEURT-20"):
        self.checkpoint = checkpoint
        self.metric = None

    def load(self, device: str) -> None:
        self.metric = evaluate.load("bleurt", module_type="metric", checkpoint=self.checkpoint)

    def compute(self, predictions: Sequence[str], references: Optional[Sequence[str]] = None) -> list[float]:
        sanitized, valid_idx = sanitize(predictions)
        result = self.metric.compute(predictions=sanitized, references=list(references))
        return [v if i in valid_idx else 0.0 for i, v in enumerate(result["scores"])]

    def unload(self) -> None:
        del self.metric
        self.metric = None


class PerplexityAdapter(MetricAdapter):
    name = "ppl"
    direction = MetricDirection.LOWER_IS_BETTER
    needs_reference = False

    def __init__(self, model_id="gpt2", batch_size=2):
        self.model_id = model_id
        self.batch_size = batch_size
        self.metric = None
        self.device = None
        self.invalid_sentinel = 1e6

    def load(self, device: str) -> None:
        self.device = device
        self.metric = evaluate.load("perplexity", module_type="metric")

    def compute(self, predictions: Sequence[str], references: Optional[Sequence[str]] = None) -> list[float]:
        sanitized, valid_idx = sanitize(predictions)
        result = self.metric.compute(
            predictions=sanitized,
            model_id=self.model_id,
            device=self.device,
            batch_size=self.batch_size,
        )
        return [
            v if i in valid_idx else self.invalid_sentinel
            for i, v in enumerate(result["perplexities"])
        ]

    def unload(self) -> None:
        del self.metric
        self.metric = None


class CoLAAdapter(MetricAdapter):
    name = "cola"
    direction = MetricDirection.HIGHER_IS_BETTER
    needs_reference = False

    def __init__(self, model="cointegrated/roberta-large-cola-krishna2020", acceptable_label="LABEL_1", batch_size=4):
        self.model = model
        self.acceptable_label = acceptable_label
        self.batch_size = batch_size
        self.pipe = None
        self.device_int = -1

    def load(self, device: str) -> None:
        self.device_int = 0 if device == "cuda" else -1
        self.pipe = pipeline("text-classification", model=self.model, device=self.device_int)

    def compute(self, predictions: Sequence[str], references: Optional[Sequence[str]] = None) -> list[float]:
        sanitized, valid_idx = sanitize(predictions)
        raw = self.pipe(sanitized, batch_size=self.batch_size, truncation=True, max_length=512)
        scores = []
        for i, out in enumerate(raw):
            if i not in valid_idx:
                scores.append(0.0)
                continue
            p = out["score"] if out["label"] == self.acceptable_label else 1.0 - out["score"]
            scores.append(p)
        return scores

    def unload(self) -> None:
        del self.pipe
        self.pipe = None


class DCRSAdapter(MetricAdapter):
    name = "dcrs"
    direction = MetricDirection.CLOSER_TO_BASELINE
    needs_reference = False

    def compute(self, predictions: Sequence[str], references: Optional[Sequence[str]] = None) -> list[float]:
        scores = []
        for t in predictions:
            t = t if isinstance(t, str) and t.strip() else "Empty."
            scores.append(textstat.dale_chall_readability_score(t))
        return scores


class CLIAdapter(MetricAdapter):
    name = "cli"
    direction = MetricDirection.CLOSER_TO_BASELINE
    needs_reference = False

    def compute(self, predictions: Sequence[str], references: Optional[Sequence[str]] = None) -> list[float]:
        scores = []
        for t in predictions:
            t = t if isinstance(t, str) and t.strip() else "Empty."
            scores.append(textstat.coleman_liau_index(t))
        return scores
