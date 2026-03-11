# Human Simulation Fidelity Evaluation

This project evaluates how faithfully AI-generated answers mimic real human interview responses in a consumer research context. It uses a LangGraph state graph to orchestrate the full evaluation pipeline — from data loading through LLM-as-judge scoring to visualization — demonstrating structured AI workflow orchestration with explicit, inspectable state at every step.

## Setup

```bash
git clone <repo-url> && cd rb_genai_eval
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Place `RB_GenAI_Datatest.xlsx` in the project root.

## Run

```bash
python main.py
```

## Output (`results/`)

| File | Description |
|---|---|
| `scores_raw.csv` | All 30 rows with dimension scores, composite, verdict, and reasons |
| `summary.json` | Aggregate metrics: means, per-person stats, verdict distribution, top failures/successes |
| `person_cards.json` | Ground-truth persona profiles extracted from real human answers |
| `figures/composite_by_person.png` | Mean composite fidelity score per person |
| `figures/dimension_heatmap.png` | Heatmap of topical/behavioral/persona/style scores by person |
| `figures/identity_match_rate.png` | Cross-person distinctiveness test results vs. 33.3% baseline |
| `figures/verdict_distribution.png` | Faithful / partly faithful / weak verdicts by person |

## Pipeline Nodes

1. **load_data** — Loads Excel, validates schema (30 rows, 3 persons)
2. **build_person_cards** — Builds ground-truth persona profiles from human answers via Groq LLM
3. **compute_similarity** — Computes cosine similarity between human and AI answer embeddings
4. **score_pairs** — LLM-as-judge scores each pair on 4 dimensions + hallucination penalty
5. **identity_match** — Tests whether each AI answer is closest to its own human (vs. other 2)
6. **aggregate** — Computes summary statistics, saves CSV/JSON outputs
7. **visualize** — Generates 4 charts

## Model

**llama-3.3-70b-versatile** via Groq API. 276 tokens/second, 86% MMLU, ~$0.01 total cost for 33 API calls. Chosen for reliable structured JSON output with strict rubrics.
