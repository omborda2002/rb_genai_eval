import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from state import EvaluationState

matplotlib.rcParams["font.family"] = "DejaVu Sans"
logger = logging.getLogger(__name__)

FIGURE_DIR = "results/figures"
DPI = 150


def visualize_node(state: EvaluationState) -> dict:
    """Generate all result figures."""
    df = pd.DataFrame(state["records"])
    identity_summary = state["identity_summary"]

    for col in ["topical", "behavioral", "persona", "style", "composite"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sns.set_style("whitegrid")

    _plot_composite_by_person(df)
    _plot_dimension_heatmap(df)
    _plot_identity_match_rate(identity_summary)
    _plot_verdict_distribution(df)

    logger.info("All figures saved to %s", FIGURE_DIR)
    return {}


def _plot_composite_by_person(df: pd.DataFrame) -> None:
    means = df.groupby("person_id")["composite"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(means.index, means.values, color=sns.color_palette("Blues_d", len(means)))
    ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=1)
    for bar in bars:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}", va="center", fontsize=10)
    ax.set_xlabel("Mean Composite Score")
    ax.set_title("Mean Composite Fidelity Score by Person")
    ax.set_xlim(0, 5.5)
    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/composite_by_person.png", dpi=DPI)
    plt.close(fig)


def _plot_dimension_heatmap(df: pd.DataFrame) -> None:
    dims = df.groupby("person_id")[["topical", "behavioral", "persona", "style"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(dims, annot=True, fmt=".2f", cmap="YlOrRd_r", vmin=1, vmax=5,
                linewidths=0.5, ax=ax)
    ax.set_title("Evaluation Dimensions by Person")
    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/dimension_heatmap.png", dpi=DPI)
    plt.close(fig)


def _plot_identity_match_rate(identity_summary: dict) -> None:
    per_person = identity_summary["per_person"]
    persons = list(per_person.keys())
    rates = [per_person[p] for p in persons]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(persons, rates, color=sns.color_palette("Blues_d", len(persons)))
    ax.axhline(y=0.333, color="red", linestyle="--", linewidth=1, label="Random baseline (33.3%)")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.0%}", ha="center", fontsize=10)
    ax.set_ylabel("Identity Match Rate")
    ax.set_title("Identity Match Rate \u2014 Does the AI Sound Like Its Own Person?")
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/identity_match_rate.png", dpi=DPI)
    plt.close(fig)


def _plot_verdict_distribution(df: pd.DataFrame) -> None:
    verdict_counts = df.groupby(["person_id", "verdict"]).size().unstack(fill_value=0)
    for v in ["faithful", "partly_faithful", "weak"]:
        if v not in verdict_counts.columns:
            verdict_counts[v] = 0
    verdict_counts = verdict_counts[["faithful", "partly_faithful", "weak"]]

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(8, 5))
    verdict_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Verdict Distribution by Person")
    ax.set_xlabel("Person")
    ax.set_ylabel("Count")
    ax.legend(title="Verdict")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/verdict_distribution.png", dpi=DPI)
    plt.close(fig)
