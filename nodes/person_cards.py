import json
import logging
import os
import time
import pandas as pd
from groq import Groq
from state import EvaluationState
from prompts import PERSON_CARD_SYSTEM, PERSON_CARD_USER

logger = logging.getLogger(__name__)
MODEL = "llama-3.3-70b-versatile"


def build_person_cards_node(state: EvaluationState) -> dict:
    """Build a ground-truth persona profile for each person from their real answers only."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    df = pd.DataFrame(state["records"])
    person_cards = {}
    errors = list(state["errors"])

    for person_id in df["person_id"].unique():
        logger.info("Building person card for %s", person_id)
        person_df = df[df["person_id"] == person_id]

        qa_block = "\n\n".join([
            f"Q: {row['question']}\nA: {row['human_answers']}"
            for _, row in person_df.iterrows()
        ])

        prompt = PERSON_CARD_USER.format(person_id=person_id, qa_block=qa_block)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": PERSON_CARD_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            card = json.loads(raw)
            person_cards[person_id] = card
            logger.info("Person card built for %s", person_id)

        except Exception as e:
            msg = f"Person card failed for {person_id}: {e}"
            logger.error(msg)
            errors.append(msg)
            person_cards[person_id] = {}

        time.sleep(2.0)

    return {"person_cards": person_cards, "errors": errors}
