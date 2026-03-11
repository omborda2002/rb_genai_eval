import json
import logging
import os
import time
import pandas as pd
from groq import Groq
from tqdm import tqdm
from state import EvaluationState
from prompts import JUDGE_SYSTEM, JUDGE_USER

logger = logging.getLogger(__name__)
MODEL = "llama-3.3-70b-versatile"

SCORE_COLS = ["topical", "behavioral", "persona", "style",
              "topical_reason", "behavioral_reason", "persona_reason", "style_reason",
              "hallucination_penalty", "hallucination_reason", "verdict", "composite"]


def _parse_score(raw: str) -> dict:
    """Strip markdown fences and parse JSON. Returns empty dict on failure."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _compute_composite(scores: dict) -> float:
    """Weighted composite score minus hallucination penalty, clamped to 0."""
    raw = (
        0.15 * scores.get("topical", 0) +
        0.30 * scores.get("behavioral", 0) +
        0.30 * scores.get("persona", 0) +
        0.25 * scores.get("style", 0) -
        scores.get("hallucination_penalty", 0)
    )
    return max(0.0, raw)


def score_pairs_node(state: EvaluationState) -> dict:
    """Run LLM-as-judge scoring on all 30 answer pairs."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    df = pd.DataFrame(state["records"])
    person_cards = state["person_cards"]
    errors = list(state["errors"])

    for col in SCORE_COLS:
        df[col] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring pairs"):
        card = person_cards.get(row["person_id"], {})
        prompt = JUDGE_USER.format(
            person_card=json.dumps(card, indent=2),
            category=row["question_category"],
            question=row["question"],
            human_answer=row["human_answers"],
            ai_answer=row["ai_answers"],
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            scores = _parse_score(response.choices[0].message.content)
            scores["composite"] = _compute_composite(scores)
            for col in SCORE_COLS:
                df.at[idx, col] = scores.get(col)

        except Exception as e:
            msg = f"Scoring failed for row {idx} ({row['person_id']}): {e}"
            logger.error(msg)
            errors.append(msg)

        time.sleep(2.0)

    logger.info("Scoring complete. Mean composite: %.2f", df["composite"].mean())
    return {"records": df.to_dict(orient="records"), "errors": errors}
