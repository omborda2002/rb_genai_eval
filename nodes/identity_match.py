import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from state import EvaluationState

logger = logging.getLogger(__name__)
EMBED_MODEL = "all-MiniLM-L6-v2"


def identity_match_node(state: EvaluationState) -> dict:
    """
    For each AI answer, check whether it is semantically closest to its
    own human's answer — compared against the other two humans' answers
    to the same question. Reuses cached human embeddings from similarity node.
    """
    df = pd.DataFrame(state["records"])
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    human_embs = np.array(state["human_embeddings"])
    ai_embs = model.encode(df["ai_answers"].tolist(), batch_size=32, convert_to_numpy=True)

    person_ids = df["person_id"].unique().tolist()
    identity_matches = []

    for i, row in df.iterrows():
        same_q_mask = df["question"] == row["question"]
        same_q_indices = df[same_q_mask].index.tolist()

        ai_emb = ai_embs[i].reshape(1, -1)
        sims = {
            df.loc[j, "person_id"]: float(cosine_similarity(ai_emb, human_embs[j].reshape(1, -1))[0][0])
            for j in same_q_indices
        }
        best_match = max(sims, key=sims.get)
        identity_matches.append(best_match == row["person_id"])

    df["identity_match"] = identity_matches

    overall_rate = sum(identity_matches) / len(identity_matches)
    per_person = {
        pid: float(df[df["person_id"] == pid]["identity_match"].mean())
        for pid in person_ids
    }

    identity_summary = {
        "overall_identity_match_rate": overall_rate,
        "per_person": per_person,
        "random_baseline": 0.333,
    }

    logger.info("Identity Match Rate: %.1f%%  (baseline: 33.3%%)", overall_rate * 100)
    return {"records": df.to_dict(orient="records"), "identity_summary": identity_summary}
