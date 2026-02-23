from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from backend.data_processor import DataProcessor, ProcessedData


@dataclass
class StabilityPlasticityResult:
    cycle: int
    timestamp: str
    feedback_samples: int
    stability_accuracy: Optional[float]
    plasticity_accuracy: Optional[float]
    base_model: Optional[str] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None


class StabilityPlasticityTracker:
    """Logs the stability–plasticity trade-off across incremental feedback cycles.

    - Stability: accuracy on a fixed, labeled gold set (hold-out)
    - Plasticity: accuracy on the newly-labeled feedback batch used in that cycle

    This is intentionally lightweight: a single CSV you can paste into README.
    """

    def __init__(
        self,
        gold_set_path: str,
        results_csv_path: str,
    ):
        self.gold_set_path = gold_set_path
        self.results_csv_path = results_csv_path
        os.makedirs(os.path.dirname(self.results_csv_path), exist_ok=True)

    def gold_set_exists(self) -> bool:
        return os.path.exists(self.gold_set_path)

    def load_gold_set(self, data_processor: DataProcessor) -> Optional[ProcessedData]:
        if not self.gold_set_exists():
            return None
        try:
            return data_processor.process_uploaded_files([self.gold_set_path], use_labels=True)
        except Exception:
            return None

    def append_result(self, result: StabilityPlasticityResult) -> None:
        file_exists = os.path.exists(self.results_csv_path)
        fieldnames = [
            "cycle",
            "timestamp",
            "feedback_samples",
            "stability_accuracy",
            "plasticity_accuracy",
            "base_model",
            "learning_rate",
            "batch_size",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]

        with open(self.results_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "cycle": result.cycle,
                    "timestamp": result.timestamp,
                    "feedback_samples": result.feedback_samples,
                    "stability_accuracy": result.stability_accuracy,
                    "plasticity_accuracy": result.plasticity_accuracy,
                    "base_model": result.base_model,
                    "learning_rate": result.learning_rate,
                    "batch_size": result.batch_size,
                    "lora_r": result.lora_r,
                    "lora_alpha": result.lora_alpha,
                    "lora_dropout": result.lora_dropout,
                }
            )

    def make_result(
        self,
        *,
        cycle: int,
        feedback_samples: int,
        stability_accuracy: Optional[float],
        plasticity_accuracy: Optional[float],
        config: Optional[Any] = None,
    ) -> StabilityPlasticityResult:
        ts = datetime.now().isoformat(timespec="seconds")
        base_model = None
        learning_rate = None
        batch_size = None
        lora_r = None
        lora_alpha = None
        lora_dropout = None

        if config is not None:
            try:
                base_model = config.model.base_model
                learning_rate = float(config.training.learning_rate)
                batch_size = int(config.training.batch_size)
                lora_r = int(config.training.lora_r)
                lora_alpha = int(config.training.lora_alpha)
                lora_dropout = float(config.training.lora_dropout)
            except Exception:
                pass

        return StabilityPlasticityResult(
            cycle=cycle,
            timestamp=ts,
            feedback_samples=feedback_samples,
            stability_accuracy=stability_accuracy,
            plasticity_accuracy=plasticity_accuracy,
            base_model=base_model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def read_results(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.results_csv_path):
            return None
        try:
            return pd.read_csv(self.results_csv_path)
        except Exception:
            return None

    @staticmethod
    def to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df is None or df.empty:
            return "(no results yet)"

        view = df.copy()
        if len(view) > max_rows:
            view = view.tail(max_rows)

        # Format floats for readability
        for col in ("stability_accuracy", "plasticity_accuracy"):
            if col in view.columns:
                view[col] = view[col].apply(
                    lambda x: "" if pd.isna(x) else f"{float(x):.3f}"
                )

        cols = [
            c
            for c in [
                "cycle",
                "feedback_samples",
                "stability_accuracy",
                "plasticity_accuracy",
                "learning_rate",
                "timestamp",
            ]
            if c in view.columns
        ]
        return view[cols].to_markdown(index=False)

    def summarize_latest(self) -> Dict[str, Any]:
        df = self.read_results()
        if df is None or df.empty:
            return {"status": "no_results"}

        latest = df.iloc[-1].to_dict()
        return {
            "status": "ok",
            "latest": latest,
            "results_csv": self.results_csv_path,
            "gold_set": self.gold_set_path,
        }
