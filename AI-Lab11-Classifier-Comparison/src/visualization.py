"""Visualization for Lab 11.

Creates:
- accuracy/precision/recall/f1 line plot
- f1 bar chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


ORDER = ["Bernoulli", "Random Forest", "Gaussian", "Decision Tree", "Multinomial", "KNeighbors"]


def plot_metrics(results: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    # Ensure consistent ordering
    names = [n for n in ORDER if n in results]

    accuracies = [results[n]["accuracy"] for n in names]
    precisions = [results[n]["precision"] for n in names]
    recalls = [results[n]["recall"] for n in names]
    f1s = [results[n]["f1"] for n in names]

    # Line graph: Accuracy/Precision/Recall/F1
    plt.figure(figsize=(16, 10))
    x = np.array(names)

    plt.plot(x, accuracies, marker="o", label="Accuracy")
    plt.plot(x, precisions, marker="o", label="Precision")
    plt.plot(x, recalls, marker="o", label="Recall")
    plt.plot(x, f1s, marker="o", label="F1")

    plt.title("Scores of Applied Classifiers")
    plt.legend()
    plt.tight_layout()

    acc_plot_path = out_dir / "accuracy_plot.png"
    plt.savefig(acc_plot_path, dpi=200)
    plt.close()

    # Bar graph: F1 scores
    plt.figure(figsize=(16, 10))
    left = np.arange(len(names)) + 1

    plt.bar(
        left,
        f1s,
        tick_label=names,
        width=0.9,
        color=["#08737f", "#00898a", "#089f8f", "#39b48e", "#64c987", "#92dc7e"],
    )

    plt.xlabel("Classifiers")
    plt.ylabel("F1 Scores")
    plt.title("F1 Scores of Applied Classifiers")
    plt.tight_layout()

    f1_plot_path = out_dir / "f1_bar_chart.png"
    plt.savefig(f1_plot_path, dpi=200)
    plt.close()

