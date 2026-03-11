import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from state import EvaluationState

logger = logging.getLogger(__name__)
EMBED_MODEL = "all-MiniLM-L6-v2"


def compute_similarity_node(state: EvaluationState) -> dict:
    """Compute cosine similarity between human and AI answers using sentence embeddings."""
    logger.info("Computing semantic similarity with %s", EMBED_MODEL)
    df = pd.DataFrame(state["records"])
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    human_embs = model.encode(df["human_answers"].tolist(), batch_size=32, convert_to_numpy=True)
    ai_embs = model.encode(df["ai_answers"].tolist(), batch_size=32, convert_to_numpy=True)

    sims = [
        float(cosine_similarity([h], [a])[0][0])
        for h, a in zip(human_embs, ai_embs)
    ]
    df["semantic_similarity"] = sims

    logger.info("Semantic similarity computed. Mean: %.3f", np.mean(sims))
    return {
        "records": df.to_dict(orient="records"),
        "human_embeddings": human_embs.tolist(),
    }
