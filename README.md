# Human Simulation Fidelity Evaluation

This project was built as part of a take-home assignment for Roland Berger. The goal is to evaluate how accurately AI-generated human simulations mimic real consumer interview responses. The pipeline implements a multi-layer fidelity framework using LangGraph for workflow orchestration, Groq-hosted LLMs for structured evaluation, and sentence embeddings for cross-person identity testing.

## Approach

The evaluation runs across three layers.

The first layer scores each AI answer against its paired human answer across four dimensions: topical alignment, behavioral fidelity, persona fidelity, and style calibration. A hallucination penalty is applied when the AI introduces preferences or values the real person never expressed.

The second layer aggregates scores per person and per question category to show where simulations consistently break down versus where they hold up.

The third layer runs an identity match test. Each AI answer is compared against all three humans' answers to the same question. A match is only counted if the AI answer is semantically closest to its own person. The random baseline for three people is 33.3%, so anything above that indicates genuine individual fidelity.

## Stack

- **LangGraph** for state graph orchestration across 7 pipeline nodes
- **Groq API** with llama-3.3-70b-versatile for LLM-as-judge scoring with structured JSON output
- **sentence-transformers** with all-MiniLM-L6-v2 for semantic similarity and identity matching
- **pandas, seaborn, matplotlib** for aggregation and visualization

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

## Output

All results are saved to the `results/` folder after the pipeline completes.

| File | Description |
|---|---|
| `scores_raw.csv` | All 30 scored pairs with dimension scores, composite score, verdict, and reasoning per dimension |
| `summary.json` | Aggregate metrics including composite means, per-person stats, verdict distribution, and top failures and successes |
| `person_cards.json` | Ground-truth persona profiles extracted from real human answers before any scoring begins |
| `figures/composite_by_person.png` | Mean composite fidelity score per person |
| `figures/dimension_heatmap.png` | Scores across topical, behavioral, persona, and style dimensions by person |
| `figures/identity_match_rate.png` | Cross-person distinctiveness test results vs. the 33.3% random baseline |
| `figures/verdict_distribution.png` | Faithful, partly faithful, and weak verdict counts by person |

## Pipeline Nodes

| Node | Role |
|---|---|
| `load_data` | Loads and validates the Excel dataset |
| `build_person_cards` | Extracts structured persona profiles from real human answers only |
| `compute_similarity` | Computes cosine similarity between human and AI answer embeddings |
| `score_pairs` | LLM-as-judge scores all 30 pairs on 4 dimensions plus hallucination penalty |
| `identity_match` | Tests whether each AI answer is closest to its own human across all three persons |
| `aggregate` | Computes summary statistics and saves all outputs |
| `visualize` | Generates the four result figures |# Human Simulation Fidelity Evaluation

This project was built as part of a take-home assignment for Roland Berger. The goal is to evaluate how accurately AI-generated human simulations mimic real consumer interview responses. The pipeline implements a multi-layer fidelity framework using LangGraph for workflow orchestration, Groq-hosted LLMs for structured evaluation, and sentence embeddings for cross-person identity testing.

## Approach

The evaluation runs across three layers.

The first layer scores each AI answer against its paired human answer across four dimensions: topical alignment, behavioral fidelity, persona fidelity, and style calibration. A hallucination penalty is applied when the AI introduces preferences or values the real person never expressed.

The second layer aggregates scores per person and per question category to show where simulations consistently break down versus where they hold up.

The third layer runs an identity match test. Each AI answer is compared against all three humans' answers to the same question. A match is only counted if the AI answer is semantically closest to its own person. The random baseline for three people is 33.3%, so anything above that indicates genuine individual fidelity.

## Stack

- **LangGraph** for state graph orchestration across 7 pipeline nodes
- **Groq API** with llama-3.3-70b-versatile for LLM-as-judge scoring with structured JSON output
- **sentence-transformers** with all-MiniLM-L6-v2 for semantic similarity and identity matching
- **pandas, seaborn, matplotlib** for aggregation and visualization

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

## Output

All results are saved to the `results/` folder after the pipeline completes.

| File | Description |
|---|---|
| `scores_raw.csv` | All 30 scored pairs with dimension scores, composite score, verdict, and reasoning per dimension |
| `summary.json` | Aggregate metrics including composite means, per-person stats, verdict distribution, and top failures and successes |
| `person_cards.json` | Ground-truth persona profiles extracted from real human answers before any scoring begins |
| `figures/composite_by_person.png` | Mean composite fidelity score per person |
| `figures/dimension_heatmap.png` | Scores across topical, behavioral, persona, and style dimensions by person |
| `figures/identity_match_rate.png` | Cross-person distinctiveness test results vs. the 33.3% random baseline |
| `figures/verdict_distribution.png` | Faithful, partly faithful, and weak verdict counts by person |

## Pipeline Nodes

| Node | Role |
|---|---|
| `load_data` | Loads and validates the Excel dataset |
| `build_person_cards` | Extracts structured persona profiles from real human answers only |
| `compute_similarity` | Computes cosine similarity between human and AI answer embeddings |
| `score_pairs` | LLM-as-judge scores all 30 pairs on 4 dimensions plus hallucination penalty |
| `identity_match` | Tests whether each AI answer is closest to its own human across all three persons |
| `aggregate` | Computes summary statistics and saves all outputs |
| `visualize` | Generates the four result figures |