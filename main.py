import os
import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from state import EvaluationState
from nodes.load_data import load_data_node
from nodes.person_cards import build_person_cards_node
from nodes.similarity import compute_similarity_node
from nodes.score_pairs import score_pairs_node
from nodes.identity_match import identity_match_node
from nodes.aggregate import aggregate_node
from nodes.visualize import visualize_node

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


def build_graph():
    graph = StateGraph(EvaluationState)

    graph.add_node("load_data", load_data_node)
    graph.add_node("build_person_cards", build_person_cards_node)
    graph.add_node("compute_similarity", compute_similarity_node)
    graph.add_node("score_pairs", score_pairs_node)
    graph.add_node("identity_match", identity_match_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("visualize", visualize_node)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "build_person_cards")
    graph.add_edge("build_person_cards", "compute_similarity")
    graph.add_edge("compute_similarity", "score_pairs")
    graph.add_edge("score_pairs", "identity_match")
    graph.add_edge("identity_match", "aggregate")
    graph.add_edge("aggregate", "visualize")
    graph.add_edge("visualize", END)

    return graph.compile()


def main():
    os.makedirs("results/figures", exist_ok=True)

    initial_state: EvaluationState = {
        "data_path": "RB_GenAI_Datatest.xlsx",
        "records": None,
        "person_cards": None,
        "human_embeddings": None,
        "identity_summary": None,
        "summary": None,
        "errors": [],
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    summary = final_state["summary"]
    identity = summary["identity_match"]

    print("\n=== FINAL RESULTS ===")
    print(f"Overall Composite Score : {summary['overall_composite_mean']:.2f} / 5.0")
    print(f"Identity Match Rate     : {identity['overall_identity_match_rate']:.1%}  (random baseline: 33.3%)")
    print(f"Verdicts → Faithful: {summary['verdict_distribution']['faithful']}  |  "
          f"Partly: {summary['verdict_distribution']['partly_faithful']}  |  "
          f"Weak: {summary['verdict_distribution']['weak']}")

    if final_state["errors"]:
        print(f"\nWarnings ({len(final_state['errors'])} non-fatal errors logged)")

    print("\nOutputs saved to ./results/")


if __name__ == "__main__":
    main()
