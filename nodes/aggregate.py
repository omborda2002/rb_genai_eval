import json
import logging
import pandas as pd
from state import EvaluationState

logger = logging.getLogger(__name__)


def aggregate_node(state: EvaluationState) -> dict:
    """Compute all summary statistics and save results to disk."""
    df = pd.DataFrame(state["records"])

    # Ensure numeric types for score columns
    for col in ["topical", "behavioral", "persona", "style", "hallucination_penalty", "composite", "semantic_similarity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    verdict_dist = df["verdict"].value_counts().to_dict()
    verdict_dist.setdefault("faithful", 0)
    verdict_dist.setdefault("partly_faithful", 0)
    verdict_dist.setdefault("weak", 0)

    per_person_composite = df.groupby("person_id")["composite"].mean().to_dict()
    per_category_composite = df.groupby("question_category")["composite"].mean().to_dict()
    per_person_dimensions = (
        df.groupby("person_id")[["topical", "behavioral", "persona", "style"]].mean()
        .round(2).to_dict(orient="index")
    )

    top_failures = (
        df.nsmallest(3, "composite")[["person_id", "question", "composite", "verdict"]]
        .to_dict(orient="records")
    )
    top_successes = (
        df.nlargest(3, "composite")[["person_id", "question", "composite", "verdict"]]
        .to_dict(orient="records")
    )

    summary = {
        "overall_composite_mean": float(df["composite"].mean()),
        "overall_semantic_similarity_mean": float(df["semantic_similarity"].mean()),
        "per_person_composite": per_person_composite,
        "per_category_composite": per_category_composite,
        "per_person_dimensions": per_person_dimensions,
        "identity_match": state["identity_summary"],
        "verdict_distribution": verdict_dist,
        "hallucination_penalty_mean": float(df["hallucination_penalty"].mean()),
        "top_failures": top_failures,
        "top_successes": top_successes,
    }

    df.to_csv("results/scores_raw.csv", index=False)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("results/person_cards.json", "w") as f:
        json.dump(state["person_cards"], f, indent=2)

    logger.info("Aggregation complete. Results saved.")
    return {"summary": summary}
