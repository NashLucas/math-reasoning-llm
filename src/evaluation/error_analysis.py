"""Error analysis and taxonomy classification engine (Phase 5 Day 31).

Classifies mathematical reasoning errors into a rigorous 7-category taxonomy:
1. Rule-based checks (evaluated first):
   - parse_error: Output empty, whitespace, malformed, or missing any valid numeric answer.
   - truncation: Output hit maximum generation tokens or cut off abruptly mid-derivation without concluding.
2. Semantic reasoning errors (evaluated via 7B instruct or heuristic classifier):
   - arithmetic_slip: Sound logical setup, but elementary computation/arithmetic mistake.
   - misread_question: Misread or hallucinated numbers, conditions, entities, or solved for the wrong quantity.
   - wrong_setup: Incorrect mathematical formulation, wrong formula, or conceptual flaw.
   - incomplete_reasoning: Stopped midway, omitted intermediate steps, or guessed answer prematurely.
   - other: Degenerate repetition, language mixing, or incoherent failure.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent, parse_numeric_value

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Rigorous error taxonomy for mathematical post-training evaluation."""

    PARSE_ERROR = "parse_error"
    TRUNCATION = "truncation"
    ARITHMETIC_SLIP = "arithmetic_slip"
    MISREAD_QUESTION = "misread_question"
    WRONG_SETUP = "wrong_setup"
    INCOMPLETE_REASONING = "incomplete_reasoning"
    OTHER = "other"


RULE_BASED_CATEGORIES = (ErrorCategory.PARSE_ERROR, ErrorCategory.TRUNCATION)
SEMANTIC_CATEGORIES = (
    ErrorCategory.ARITHMETIC_SLIP,
    ErrorCategory.MISREAD_QUESTION,
    ErrorCategory.WRONG_SETUP,
    ErrorCategory.INCOMPLETE_REASONING,
    ErrorCategory.OTHER,
)
ALL_ERROR_CATEGORIES = tuple(c.value for c in ErrorCategory)


@dataclass
class ErrorClassification:
    """Detailed error classification record for a problem attempt."""

    problem_id: str
    question: str
    gold_answer: str
    generated_text: str
    extracted_answer: Optional[str]
    is_correct: bool
    is_error: bool
    category: Optional[ErrorCategory]  # None if correct
    classification_source: str  # 'rule_based', 'llm_classifier', 'heuristic_fallback', 'ground_truth'
    rationale: str
    gold_solution: Optional[str] = None
    extraction_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "generated_text": self.generated_text,
            "extracted_answer": self.extracted_answer,
            "is_correct": self.is_correct,
            "is_error": self.is_error,
            "category": self.category.value if self.category else None,
            "classification_source": self.classification_source,
            "rationale": self.rationale,
            "gold_solution": self.gold_solution,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
        }


# ==============================================================================
# Rule-Based Error Detection
# ==============================================================================


def check_truncation(
    generated_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 512,
) -> Tuple[bool, str]:
    """Determine whether an output suffered from token truncation.

    Rules:
    1. Explicit finish reason indicating length/token limit.
    2. Explicit metadata flag indicating truncation.
    3. Output text ends abruptly without terminal punctuation, closing brackets,
       or canonical answer format ('#### <number>'), especially when near length limit.
    4. Unclosed LaTeX boxed expression: '\\boxed{' without matching '}'.
    """
    if metadata:
        finish_reason = str(metadata.get("finish_reason", "")).lower()
        if finish_reason in ("length", "max_tokens", "token_limit"):
            return True, f"Explicit finish_reason='{finish_reason}' reported by engine"

        if metadata.get("is_truncated") is True:
            return True, "Metadata explicitly flagged is_truncated=True"

    if not generated_text or not generated_text.strip():
        return False, "Empty text is treated as a parse error, not truncation"

    cleaned = generated_text.rstrip()

    # If canonical #### delimiter is present with an answer at the end, it was not truncated
    if re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", cleaned):
        return False, "Canonical final answer present; not truncated"

    # Check unclosed LaTeX \boxed{...
    if r"\boxed{" in cleaned:
        # Check matching braces
        idx = cleaned.rfind(r"\boxed{")
        sub = cleaned[idx + len(r"\boxed{") :]
        if sub.count("}") < 1:
            return True, "Unclosed LaTeX \\boxed{ expression indicates truncation"

    # Check abrupt grammatical cutoffs (ends with arithmetic operator, conjunction, or incomplete sentence)
    abrupt_endings = (
        r"[-+*/=]\s*$",
        r"\b(?:and|or|then|so|because|with|plus|minus|times|divided by|equal to|equals)\s*$",
        r"\b(?:the|a|an|of|to|in|for|is|are|was|were)\s*$",
        r"\.{3}\s*$",  # Trailing ellipsis
    )
    for pattern in abrupt_endings:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return True, f"Abrupt grammatical cutoff matching pattern: {pattern}"

    # Approximate token count check (1 word ~= 1.3 tokens)
    word_count = len(cleaned.split())
    estimated_tokens = int(word_count * 1.3)
    if estimated_tokens >= max_new_tokens - 10:
        # Near token limit and no terminal punctuation
        if not re.search(r"[.!?\n]\s*$", cleaned):
            return True, f"Near token limit ({estimated_tokens} ~ {max_new_tokens}) without terminal punctuation"

    return False, "No truncation signals detected"


def check_parse_failure(
    generated_text: str,
    extracted_answer: Optional[str] = None,
    extraction_method: Optional[str] = None,
) -> Tuple[bool, str]:
    """Determine whether an output suffered from answer extraction / parse failure.

    Rules:
    1. Empty or whitespace-only generation.
    2. Extracted answer is None or extraction method is 'none'.
    3. Extracted answer cannot be parsed into a numeric value.
    """
    if not generated_text or not generated_text.strip():
        return True, "Empty or whitespace-only model generation"

    if extracted_answer is None or extraction_method == "none":
        ans, method = extract_answer(generated_text)
        if ans is None or method == "none":
            return True, "No numeric answer could be extracted from generation"
        extracted_answer = ans

    # Verify extracted answer parses into a numeric scalar
    val = parse_numeric_value(extracted_answer)
    if val is None:
        return True, f"Extracted answer '{extracted_answer}' cannot be parsed into numeric value"

    return False, "Parse successful"


# ==============================================================================
# Semantic Error Classification Prompt & Heuristics
# ==============================================================================

ERROR_CLASSIFIER_SYSTEM_PROMPT = """You are an expert mathematical error analysis evaluator.
Your task is to analyze why a model's generated mathematical reasoning produced an incorrect answer, and classify the error into exactly ONE category from the rubric below.

Error Categories:
1. arithmetic_slip:
   - The conceptual setup, equations, and mathematical logic are completely sound and correct.
   - The mistake is solely an arithmetic or elementary computation error (e.g. 7 * 8 = 54, 15 + 28 = 42, 100 / 4 = 20).
2. misread_question:
   - The model misread numbers, misinterpreted constraints, confused entities, or solved for the wrong quantity asked in the question (e.g. solved for Bob instead of Alice, or used 20 instead of 30 from the prompt).
3. wrong_setup:
   - The conceptual mathematical approach, formula, or equation setup is fundamentally incorrect (e.g. multiplied instead of divided, inverted a ratio, used the wrong percentage formula, flawed geometric relation).
4. incomplete_reasoning:
   - The derivation started in the right direction but stopped midway, skipped critical steps, or guessed the final answer without working through the full chain of logic.
5. other:
   - The error does not fit the above categories (e.g. degenerate repetition, language mixing, incoherent rambling, completely ungrounded hallucinations).

Respond strictly in the following format:
Classification: <category>
Rationale: <1-2 concise sentences explaining the primary root cause of the error>
"""


def format_error_classification_prompt(
    question: str,
    gold_answer: str,
    generated_text: str,
    extracted_answer: Optional[str],
    gold_solution: Optional[str] = None,
) -> str:
    """Format prompt for 7B instruct semantic error classification."""
    prompt = f"""Problem Statement:
{question}

Ground Truth Target:
Final Answer: {gold_answer}"""

    if gold_solution:
        prompt += f"\nReference Solution: {gold_solution}"

    prompt += f"""

Model Output:
{generated_text}

Extracted Model Answer:
{extracted_answer if extracted_answer is not None else 'None'}

Please classify the error into exactly one of: arithmetic_slip, misread_question, wrong_setup, incomplete_reasoning, other.
"""
    return prompt


def parse_llm_classification(response_text: str) -> Tuple[ErrorCategory, str]:
    """Parse classification and rationale from 7B instruct output."""
    if not response_text:
        return ErrorCategory.OTHER, "Empty response from classifier model"

    # Match 'Classification: <category>'
    match = re.search(r"Classification:\s*([a-z_]+)", response_text, re.IGNORECASE)
    category_str = match.group(1).lower() if match else ""

    rationale_match = re.search(r"Rationale:\s*([^\n\r]+(?:\n[^\n\r]+)?)", response_text, re.IGNORECASE)
    rationale = rationale_match.group(1).strip() if rationale_match else response_text.strip()

    # Map to ErrorCategory enum
    for cat in SEMANTIC_CATEGORIES:
        if cat.value == category_str:
            return cat, rationale

    # Fallback search for any of the 5 semantic category keywords
    for cat in SEMANTIC_CATEGORIES:
        if re.search(r"\b" + re.escape(cat.value) + r"\b", response_text, re.IGNORECASE):
            return cat, rationale

    return ErrorCategory.OTHER, f"Unrecognized category '{category_str}'; classified as other"


def check_equation_arithmetic_error(text: str) -> Optional[str]:
    """Check if any explicit arithmetic equation 'X op Y = Z' in the text contains a calculation error."""
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)", text):
        try:
            x = float(m.group(1))
            op = m.group(2)
            y = float(m.group(3))
            z = float(m.group(4))
            if op == "+":
                expected = x + y
            elif op == "-":
                expected = x - y
            elif op == "*":
                expected = x * y
            elif op == "/":
                expected = x / y if y != 0 else None
            else:
                expected = None

            if expected is not None and not math.isclose(expected, z, rel_tol=1e-4, abs_tol=1e-4):
                return f"Calculation mistake in equation '{m.group(0)}': expected {expected}, got {z}"
        except Exception:
            continue
    return None


def classify_semantic_heuristic(
    question: str,
    gold_answer: str,
    generated_text: str,
    extracted_answer: Optional[str],
    gold_solution: Optional[str] = None,
) -> Tuple[ErrorCategory, str]:
    """Deterministic heuristic error classifier for testing, CI, and local CPU runs."""
    text_lower = generated_text.lower()

    # 1. Check for degenerate / other errors
    words = text_lower.split()
    if len(words) > 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return ErrorCategory.OTHER, "Degenerate repetition detected in generated reasoning"

    # 2. Check for explicit arithmetic equation errors (e.g. 15 + 12 = 28)
    calc_err = check_equation_arithmetic_error(generated_text)
    if calc_err:
        return ErrorCategory.ARITHMETIC_SLIP, calc_err

    # 3. Check for incomplete reasoning (minimal or no preceding steps before ####)
    text_before_answer = re.split(r"####", generated_text)[0].strip()
    has_arithmetic_eq = bool(re.search(r"\d+\s*[-+*/=]\s*\d+", text_lower))
    if len(text_before_answer) < 35 and not has_arithmetic_eq:
        return (
            ErrorCategory.INCOMPLETE_REASONING,
            "Answer provided with minimal or no preceding derivation steps",
        )

    # 4. Check for misread question
    if any(k in text_lower for k in ("instead of", "misunderstood", "wrong quantity", "confused")):
        return ErrorCategory.MISREAD_QUESTION, "Question constraint or quantity misread"

    # Check if prompt numbers were misread in premise setup (before any '=')
    premise_text = text_before_answer.split("=")[0] if "=" in text_before_answer else text_before_answer
    q_nums = set(re.findall(r"\b\d+\b", question))
    premise_nums = set(re.findall(r"\b\d+\b", premise_text))
    foreign_premise_nums = premise_nums - q_nums
    if foreign_premise_nums and any(
        kw in premise_text.lower() for kw in ("misread", "assume", "let", "given that", "starts with")
    ):
        return ErrorCategory.MISREAD_QUESTION, f"Model introduced premise numbers not in prompt: {foreign_premise_nums}"

    # 5. Check for arithmetic slip if equations present and predicted answer is close to gold
    gold_val = parse_numeric_value(gold_answer)
    pred_val = parse_numeric_value(extracted_answer)
    eq_count = text_lower.count("=")
    if eq_count >= 1 and gold_val is not None and pred_val is not None:
        if abs(gold_val - pred_val) <= max(10.0, 0.25 * abs(gold_val)):
            return (
                ErrorCategory.ARITHMETIC_SLIP,
                f"Equations present ({eq_count} '='); predicted {pred_val} close to gold {gold_val}",
            )

    # 6. Default semantic failure to wrong_setup
    return (
        ErrorCategory.WRONG_SETUP,
        "Mathematical formulation or operation setup diverged from problem requirements",
    )


# ==============================================================================
# Unified Classifier Interface
# ==============================================================================


class ErrorClassifier:
    """Unified mathematical error classifier orchestrating rules, LLM judge, and heuristic fallbacks."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_mock: bool = False,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        backend: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path or "deepseek-ai/deepseek-math-7b-instruct"
        self.use_mock = use_mock
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.backend = backend
        self._llm_pipeline = None

        if not self.use_mock and self.backend is None:
            self._init_backend()

    def _init_backend(self) -> None:
        """Initialize Hugging Face text-generation pipeline if hardware and packages permit."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            if not torch.cuda.is_available() and self.device != "cpu":
                logger.info("CUDA unavailable; running ErrorClassifier in mock/heuristic mode.")
                self.use_mock = True
                return

            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            logger.info("Loading ErrorClassifier model: %s on %s", self.model_name_or_path, device_map)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self._llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
            )
        except Exception as e:
            logger.warning("Could not initialize HF pipeline for ErrorClassifier (%s); using mock/heuristics.", e)
            self.use_mock = True

    def classify_single(
        self,
        question: str,
        gold_answer: str,
        generated_text: str,
        extracted_answer: Optional[str] = None,
        extraction_method: Optional[str] = None,
        gold_solution: Optional[str] = None,
        problem_id: str = "problem",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ErrorClassification:
        """Classify a single model attempt through the tiered rule-then-semantic pipeline."""
        metadata = dict(metadata or {})

        # Extract answer if not provided
        if extracted_answer is None:
            extracted_answer, method = extract_answer(generated_text)
            if extraction_method is None:
                extraction_method = method

        # Check equivalence with gold target
        is_corr = is_equivalent(extracted_answer, gold_answer)

        # 1. Correct Output: Not an error!
        if is_corr:
            return ErrorClassification(
                problem_id=problem_id,
                question=question,
                gold_answer=gold_answer,
                generated_text=generated_text,
                extracted_answer=extracted_answer,
                is_correct=True,
                is_error=False,
                category=None,
                classification_source="ground_truth",
                rationale="Prediction matches gold ground-truth answer.",
                gold_solution=gold_solution,
                extraction_method=extraction_method,
                metadata=metadata,
            )

        # 2. Rule 1: Truncation Check (runs first on failed attempts)
        is_trunc, trunc_reason = check_truncation(
            generated_text=generated_text,
            metadata=metadata,
            max_new_tokens=self.max_new_tokens,
        )
        if is_trunc:
            return ErrorClassification(
                problem_id=problem_id,
                question=question,
                gold_answer=gold_answer,
                generated_text=generated_text,
                extracted_answer=extracted_answer,
                is_correct=False,
                is_error=True,
                category=ErrorCategory.TRUNCATION,
                classification_source="rule_based",
                rationale=trunc_reason,
                gold_solution=gold_solution,
                extraction_method=extraction_method,
                metadata=metadata,
            )

        # 3. Rule 2: Parse Failure Check
        is_parse_err, parse_reason = check_parse_failure(
            generated_text=generated_text,
            extracted_answer=extracted_answer,
            extraction_method=extraction_method,
        )
        if is_parse_err:
            return ErrorClassification(
                problem_id=problem_id,
                question=question,
                gold_answer=gold_answer,
                generated_text=generated_text,
                extracted_answer=extracted_answer,
                is_correct=False,
                is_error=True,
                category=ErrorCategory.PARSE_ERROR,
                classification_source="rule_based",
                rationale=parse_reason,
                gold_solution=gold_solution,
                extraction_method=extraction_method,
                metadata=metadata,
            )

        # 4. Semantic Reasoning Errors: 7B Instruct / Dev Model / Heuristic Fallback
        if self.backend is not None:
            prompt = format_error_classification_prompt(
                question=question,
                gold_answer=gold_answer,
                generated_text=generated_text,
                extracted_answer=extracted_answer,
                gold_solution=gold_solution,
            )
            response_text = self.backend(prompt)
            cat, rationale = parse_llm_classification(response_text)
            source = "llm_classifier"
        elif not self.use_mock and self._llm_pipeline is not None:
            prompt = format_error_classification_prompt(
                question=question,
                gold_answer=gold_answer,
                generated_text=generated_text,
                extracted_answer=extracted_answer,
                gold_solution=gold_solution,
            )
            messages = [
                {"role": "system", "content": ERROR_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                out = self._llm_pipeline(messages)
                response_text = out[0]["generated_text"][-1]["content"]
                cat, rationale = parse_llm_classification(response_text)
                source = "llm_classifier"
            except Exception as e:
                logger.warning("LLM inference failed (%s); falling back to heuristic.", e)
                cat, rationale = classify_semantic_heuristic(
                    question=question,
                    gold_answer=gold_answer,
                    generated_text=generated_text,
                    extracted_answer=extracted_answer,
                    gold_solution=gold_solution,
                )
                source = "heuristic_fallback"
        else:
            cat, rationale = classify_semantic_heuristic(
                question=question,
                gold_answer=gold_answer,
                generated_text=generated_text,
                extracted_answer=extracted_answer,
                gold_solution=gold_solution,
            )
            source = "heuristic_fallback"

        return ErrorClassification(
            problem_id=problem_id,
            question=question,
            gold_answer=gold_answer,
            generated_text=generated_text,
            extracted_answer=extracted_answer,
            is_correct=False,
            is_error=True,
            category=cat,
            classification_source=source,
            rationale=rationale,
            gold_solution=gold_solution,
            extraction_method=extraction_method,
            metadata=metadata,
        )

    def classify_records(
        self,
        records: Sequence[Dict[str, Any]],
        limit: Optional[int] = None,
    ) -> List[ErrorClassification]:
        """Classify a list of generation/evaluation records."""
        items = list(records[:limit]) if limit else list(records)
        classifications: List[ErrorClassification] = []

        for rec in items:
            gen_text = rec.get("generated_text")
            if gen_text is None and "generated_texts" in rec and rec["generated_texts"]:
                gen_text = rec["generated_texts"][0]
            if gen_text is None:
                gen_text = rec.get("completion", "")

            ext_ans = rec.get("extracted_answer")
            if ext_ans is None and "extracted_answers" in rec and rec["extracted_answers"]:
                ext_ans = rec["extracted_answers"][0]

            ext_meth = rec.get("extraction_method")
            if ext_meth is None and "extraction_methods" in rec and rec["extraction_methods"]:
                ext_meth = rec["extraction_methods"][0]

            classification = self.classify_single(
                question=rec["question"],
                gold_answer=rec.get("gold_answer", rec.get("answer", "")),
                generated_text=gen_text,
                extracted_answer=ext_ans,
                extraction_method=ext_meth,
                gold_solution=rec.get("gold_solution", rec.get("solution")),
                problem_id=str(rec.get("problem_id", rec.get("id", len(classifications)))),
                metadata=rec.get("metadata", {}),
            )
            classifications.append(classification)

        return classifications


# ==============================================================================
# Summary & Statistics Aggregation
# ==============================================================================


def compute_error_distribution(classifications: Sequence[ErrorClassification]) -> Dict[str, Any]:
    """Compute aggregate distribution across error categories and sources."""
    total_evaluated = len(classifications)
    correct_count = sum(1 for c in classifications if c.is_correct)
    error_count = sum(1 for c in classifications if c.is_error)

    counts: Dict[str, int] = {cat: 0 for cat in ALL_ERROR_CATEGORIES}
    sources: Dict[str, int] = {}

    for c in classifications:
        if c.is_error and c.category:
            counts[c.category.value] = counts.get(c.category.value, 0) + 1
            sources[c.classification_source] = sources.get(c.classification_source, 0) + 1

    percentages: Dict[str, float] = {}
    for cat, cnt in counts.items():
        percentages[cat] = round((cnt / error_count * 100.0) if error_count > 0 else 0.0, 2)

    rule_based_total = sum(counts[cat.value] for cat in RULE_BASED_CATEGORIES)
    semantic_total = sum(counts[cat.value] for cat in SEMANTIC_CATEGORIES)

    return {
        "total_evaluated": total_evaluated,
        "correct_count": correct_count,
        "error_count": error_count,
        "accuracy": round((correct_count / total_evaluated * 100.0) if total_evaluated > 0 else 0.0, 2),
        "error_rate": round((error_count / total_evaluated * 100.0) if total_evaluated > 0 else 0.0, 2),
        "category_counts": counts,
        "category_percentages": percentages,
        "rule_based_count": rule_based_total,
        "rule_based_percentage": round((rule_based_total / error_count * 100.0) if error_count > 0 else 0.0, 2),
        "semantic_count": semantic_total,
        "semantic_percentage": round((semantic_total / error_count * 100.0) if error_count > 0 else 0.0, 2),
        "sources": sources,
    }


def generate_error_markdown_report(
    summary: Dict[str, Any],
    classifications: Sequence[ErrorClassification],
    stage_name: str = "SFT Round 1",
) -> str:
    """Generate Markdown error analysis report for reports/error_analysis_sft.md."""
    lines = [
        f"# Error Analysis Report — {stage_name}",
        "",
        "## 1. Executive Summary",
        f"- **Total Problems Evaluated:** {summary['total_evaluated']}",
        f"- **Correct Solutions:** {summary['correct_count']} ({summary['accuracy']}%)",
        f"- **Total Errors Analyzed:** {summary['error_count']} ({summary['error_rate']}%)",
        f"- **Rule-Based Failures:** {summary['rule_based_count']} ({summary['rule_based_percentage']}%)",
        f"- **Semantic Reasoning Errors:** {summary['semantic_count']} ({summary['semantic_percentage']}%)",
        "",
        "## 2. Error Category Breakdown",
        "",
        "| Category | Type | Count | % of Errors | Description |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ]

    descriptions = {
        "parse_error": "Output empty, unparseable, or missing numeric final answer",
        "truncation": "Output exceeded maximum length without terminating",
        "arithmetic_slip": "Correct mathematical setup, but arithmetic calculation error",
        "misread_question": "Misread numbers, entities, or solved for wrong quantity",
        "wrong_setup": "Incorrect formula, equation modeling, or conceptual approach",
        "incomplete_reasoning": "Derivation omitted steps or guessed answer prematurely",
        "other": "Degenerate repetition, language mixing, or incoherent failure",
    }

    for cat in ALL_ERROR_CATEGORIES:
        is_rule = cat in [r.value for r in RULE_BASED_CATEGORIES]
        cat_type = "Rule-Based" if is_rule else "Semantic"
        cnt = summary["category_counts"].get(cat, 0)
        pct = summary["category_percentages"].get(cat, 0.0)
        desc = descriptions.get(cat, "")
        lines.append(f"| `{cat}` | {cat_type} | {cnt} | {pct:.1f}% | {desc} |")

    lines.extend([
        "",
        "## 3. Representative Examples per Category",
        "",
    ])

    # Group up to 2 examples per category
    examples_by_cat: Dict[str, List[ErrorClassification]] = {cat: [] for cat in ALL_ERROR_CATEGORIES}
    for c in classifications:
        if c.is_error and c.category:
            if len(examples_by_cat[c.category.value]) < 2:
                examples_by_cat[c.category.value].append(c)

    for cat in ALL_ERROR_CATEGORIES:
        examples = examples_by_cat[cat]
        lines.append(f"### Category: `{cat}`")
        if not examples:
            lines.append("*No errors observed in this category.*")
            lines.append("")
            continue

        for idx, ex in enumerate(examples, 1):
            lines.append(f"**Example {idx} (ID: {ex.problem_id})**")
            lines.append(f"- **Question:** {ex.question}")
            lines.append(f"- **Gold Answer:** `{ex.gold_answer}`")
            lines.append(f"- **Extracted Answer:** `{ex.extracted_answer}`")
            lines.append(f"- **Rationale:** {ex.rationale}")
            lines.append("```text")
            lines.append(ex.generated_text[:300] + ("..." if len(ex.generated_text) > 300 else ""))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)
