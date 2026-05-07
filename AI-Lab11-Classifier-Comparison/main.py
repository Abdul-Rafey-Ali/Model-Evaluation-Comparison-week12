"""Entry point for Lab 11 classifier comparison.

Run:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_preprocessing import load_and_prepare
from src.models import train_and_evaluate
from src.visualization import plot_metrics


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = repo_root / "data" / "dataset.csv"
    target_col = "loan_status"

    df, X, y = load_and_prepare(data_path=data_path, target_col=target_col)

    results = train_and_evaluate(X, y)

    # Save results.txt
    result_text = [
        f"Dataset rows: {len(df)}",
        f"X shape: {X.shape}",
        f"y shape: {y.shape}",
        "",
        "Metrics (weighted where applicable):",
    ]

    for name, m in results.items():
        result_text.append(
            f"{name}: "
            f"accuracy={m['accuracy']:.4f}, "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, "
            f"f1={m['f1']:.4f}"
        )

    (out_dir / "results.txt").write_text("\n".join(result_text), encoding="utf-8")

    # Plots
    plot_metrics(results, out_dir=out_dir)

    # Console summary
    best = max(results.items(), key=lambda kv: kv[1]["f1"])
    print("Best model by weighted F1:", best[0])
    print("F1:", best[1]["f1"])
    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    # Ensure consistent randomness for models that use it
    np.random.seed(42)
    main()

