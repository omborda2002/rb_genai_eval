import logging
import pandas as pd
from state import EvaluationState

logger = logging.getLogger(__name__)


def load_data_node(state: EvaluationState) -> dict:
    """Load the Excel dataset and validate structure."""
    logger.info("Loading dataset from %s", state["data_path"])
    df = pd.read_excel(state["data_path"], sheet_name="answer_pairs")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in ["question_category", "question", "person_id", "human_answers", "ai_answers"]:
        df[col] = df[col].astype(str).str.strip()

    assert len(df) == 30, f"Expected 30 rows, got {len(df)}"
    assert df["person_id"].nunique() == 3, "Expected exactly 3 unique person IDs"

    logger.info("Loaded %d rows for persons: %s", len(df), df["person_id"].unique().tolist())
    return {"records": df.to_dict(orient="records")}
