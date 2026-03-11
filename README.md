# Human Simulation Fidelity Evaluation

Evaluation pipeline for assessing how accurately AI-generated human simulations mimic real consumer interview responses. Built as part of a GenAI take-home assignment, the project implements a multi-layer fidelity framework using LangGraph for workflow orchestration, Groq-hosted LLMs for structured evaluation, and sentence embeddings for cross-person identity testing.

## Approach

The pipeline runs three evaluation layers:

- **Pairwise scoring** — each AI answer is scored against its human counterpart across four dimensions: topical alignment, behavioral fidelity, persona fidelity, and style calibration, with a hallucination penalty
- **Dimension aggregation** — scores are rolled up per person and per question category to surface where simulations break down
- **Identity match test** — each AI answer is compared against all three humans' answers to the same question; a match is only counted if the AI is closest to its own person (random baseline: 33.3%)

## Stack

- **LangGraph** — state graph orchestration across 7 pipeline nodes
- **Groq API** (`llama-3.3-70b-versatile`) — LLM-as-judge scoring with structured JSON output
- **sentence-transformers** (`all-MiniLM-L6-v2`) — semantic similarity and identity match
- **pandas / seaborn / matplotlib** — aggregation and visualization

## Setup

```bash
git clone https://github.com/omborda2002/rb_genai_eval.git
cd rb_genai_eval
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Place `RB_GenAI_Datatest.xlsx` in the project root, then run:

```bash
python main.py
```

## Output (`results/`)

| File | Description |
|---|---|
| `scores_raw.csv` | All 30 scored pairs with dimension scores, composite, verdict, and reasoning |
| `summary.json` | Aggregate metrics: composite means, per-person stats, verdict distribution, top failures and successes |
| `person_cards.json` | Ground-truth persona profiles extracted from real human answers before scoring |
| `figures/composite_by_person.png` | Mean composite fidelity score per person |
| `figures/dimension_heatmap.png` | Scores across topical / behavioral / persona / style by person |
| `figures/identity_match_rate.png` | Cross-person distinctiveness test vs. 33.3% random baseline |
| `figures/verdict_distribution.png` | Faithful / partly faithful / weak verdict distribution by person |

## Pipeline Nodes

| Node | Role |
|---|---|
| `load_data` | Loads and validates the Excel dataset |
| `build_person_cards` | Extracts structured persona profiles from real human answers only |
| `compute_similarity` | Computes cosine similarity between human and AI answer embeddings |
| `score_pairs` | LLM-as-judge scores all 30 pairs on 4 dimensions plus hallucination penalty |
| `identity_match` | Tests cross-person distinctiveness — does each AI answer sound like its own human? |
| `aggregate` | Computes summary statistics and saves all outputs |
| `visualize` | Generates the four result figures |