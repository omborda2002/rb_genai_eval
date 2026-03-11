PERSON_CARD_SYSTEM = """You are a consumer research analyst. You will be given all interview \
answers from a single real person. Extract a structured profile based ONLY on what they explicitly \
said. Do not infer, assume, or add anything beyond the evidence in their words.
Respond with valid JSON only. No preamble, no explanation, no markdown fences."""

PERSON_CARD_USER = """Here are all answers from {person_id}:

{qa_block}

Return a JSON object with exactly these keys:
{{
  "purchase_channels": "where and how they shop",
  "brand_involvement": "high/medium/low + one sentence of evidence",
  "decision_drivers": ["list of what actually drives choices: price, convenience, ingredients, habit, social, etc."],
  "ingredient_awareness": "yes/somewhat/no + one sentence of evidence",
  "influence_sources": "what or who influences them to try new products",
  "emotional_tone": "how they communicate: casual/precise/dismissive/enthusiastic/uncertain",
  "certainty_level": "high/medium/low + one sentence of evidence",
  "key_phrases": ["3-5 actual verbatim phrases from their answers that best characterise this person"]
}}"""

JUDGE_SYSTEM = """You are a strict expert evaluator assessing how accurately an AI-generated answer \
mimics a specific real human's response in a market research context. You have the person's full \
persona card built from their real answers.

Score based on evidence only. Do not reward the AI for sounding polished or thorough. \
Penalise it for anything not grounded in what this specific person actually said.
Respond with valid JSON only. No preamble, no explanation, no markdown fences."""

JUDGE_USER = """PERSONA CARD:
{person_card}

QUESTION CATEGORY: {category}
QUESTION: {question}

HUMAN ANSWER:
{human_answer}

AI ANSWER:
{ai_answer}

Score the AI answer and return JSON with exactly this structure:
{{
  "topical": <1-5>,
  "topical_reason": "<one sentence citing specific evidence>",
  "behavioral": <1-5>,
  "behavioral_reason": "<one sentence citing specific evidence>",
  "persona": <1-5>,
  "persona_reason": "<one sentence citing specific evidence>",
  "style": <1-5>,
  "style_reason": "<one sentence citing specific evidence>",
  "hallucination_penalty": <0, 1, or 2>,
  "hallucination_reason": "<what was fabricated or injected, or 'none'>",
  "verdict": "<faithful|partly_faithful|weak>"
}}

Scoring guide:
- topical: Does the AI answer the same question with the same decision context?
- behavioral: Does it preserve the same buying logic, priorities and trade-offs?
- persona: Does it feel consistent with who this person is across all their answers?
- style: Does it match how much the person cares and how certain they are? (Penalise over-elaboration)
- hallucination_penalty: 0=nothing invented, 1=minor unsupported addition, 2=clear fabrication of brand/preference/value"""
