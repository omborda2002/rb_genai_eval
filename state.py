from typing import TypedDict, Optional

class EvaluationState(TypedDict):
    data_path: str
    records: Optional[list[dict]]
    person_cards: Optional[dict]
    human_embeddings: Optional[list[list[float]]]
    identity_summary: Optional[dict]
    summary: Optional[dict]
    errors: list[str]
