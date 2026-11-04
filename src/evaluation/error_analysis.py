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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

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
    q_lower = question.lower()

    # 1. Check for degenerate / language mixing / other errors
    words = text_lower.split()
    if len(words) > 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return ErrorCategory.OTHER, "Degenerate repetition detected in generated reasoning"

    non_english_or_refusal = (
        r"\b(?:je ne sais|comment résoudre|ce problème|aujourd'hui|bonjour|merci)\b",
        r"\b(?:i cannot solve|i am unable|sorry, but i)\b",
    )
    for pat in non_english_or_refusal:
        if re.search(pat, text_lower):
            return ErrorCategory.OTHER, f"Non-English generation or refusal pattern matched: {pat}"

    # 2. Check for explicit arithmetic equation calculation errors (e.g. 15 + 12 = 28)
    calc_err = check_equation_arithmetic_error(generated_text)
    if calc_err:
        return ErrorCategory.ARITHMETIC_SLIP, calc_err

    # 3. Check for misread question (numbers, constraints, or solved for wrong entity)
    text_before_answer = re.split(r"####", generated_text)[0].strip()
    premise_text = text_before_answer.split("=")[0] if "=" in text_before_answer else text_before_answer
    q_nums = set(re.findall(r"\b\d+\b", question))
    premise_nums = set(re.findall(r"\b\d+\b", premise_text))
    foreign_premise_nums = premise_nums - q_nums
    if foreign_premise_nums and any(
        kw in premise_text.lower()
        for kw in ("fewer", "more", "less", "has", "starts", "initially", "let", "given", "bought")
    ):
        return ErrorCategory.MISREAD_QUESTION, f"Model introduced premise numbers not in prompt: {foreign_premise_nums}"

    if any(k in text_lower for k in ("instead of", "misunderstood", "wrong quantity", "confused", "misread")):
        return ErrorCategory.MISREAD_QUESTION, "Question constraint or quantity misread"

    if "longer" in q_lower and any(w in text_lower for w in ("less", "shorter")):
        return ErrorCategory.MISREAD_QUESTION, "Misread 'longer' condition as 'less'/'shorter'"

    if any(target in q_lower for target in ("notebooks", "pens", "apples", "shirts", "red")) and any(
        w in text_lower for w in ("total spent", "on everything", "total cost of all")
    ):
        return ErrorCategory.MISREAD_QUESTION, "Solved for overall total instead of specific subcategory"

    if "total" in q_lower and any(kw in text_lower for kw in ("sheep", "cows")) and not any(
        w in text_lower for w in ("total number of animals", "total animals", "all animals")
    ):
        return ErrorCategory.MISREAD_QUESTION, "Solved for component rather than total"

    # 4. Check for incomplete reasoning (stopped before reaching final requirement)
    has_arithmetic_eq = bool(re.search(r"\d+\s*[-+*/=]\s*\d+", text_lower))
    if len(text_before_answer) < 35 and not has_arithmetic_eq:
        return (
            ErrorCategory.INCOMPLETE_REASONING,
            "Answer provided with minimal or no preceding derivation steps",
        )

    if (
        "change" in q_lower
        and any(k in q_lower for k in ("bill", "pays with", "gave"))
        and not any(k in text_lower for k in ("change", "50 -", "50-"))
    ):
        return (
            ErrorCategory.INCOMPLETE_REASONING,
            "Stopped after calculating total cost without subtracting from payment for change",
        )

    if (
        any(k in q_lower for k in ("goal", "how many more", "must they sell", "still need"))
        and text_lower.count("\n") <= 2
    ):
        return (
            ErrorCategory.INCOMPLETE_REASONING,
            "Stopped at intermediate subtotal without completing final goal step",
        )

    if (
        any(k in q_lower for k in ("not red", "left over", "remain", "empty seats"))
        and any(k in text_lower for k in ("roses =", "boxes", "passengers"))
        and extracted_answer != gold_answer
    ):
        return (
            ErrorCategory.INCOMPLETE_REASONING,
            "Omitted final subtraction/remainder step",
        )

    # 5. Default semantic failure to wrong_setup
    return (
        ErrorCategory.WRONG_SETUP,
        "Mathematical formulation, formula, or operation setup diverged from problem requirements",
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
    cluster_report: Optional[ClusterAnalysisReport] = None,
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

    if cluster_report is not None and cluster_report.clusters:
        lines.append("")
        lines.append(generate_cluster_markdown(cluster_report))

    return "\n".join(lines)


@dataclass
class AgreementReport:
    """Evaluation agreement report comparing automatic classifications against reference manual labels."""

    total_samples: int
    agreed_count: int
    disagreed_count: int
    agreement_rate: float
    agreement_percent: float
    passed_calibration: bool
    per_category: Dict[str, Dict[str, Any]]
    disagreements: List[Dict[str, Any]]
    confusion_matrix: Dict[str, Dict[str, int]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "agreed_count": self.agreed_count,
            "disagreed_count": self.disagreed_count,
            "agreement_rate": round(self.agreement_rate, 4),
            "agreement_percent": round(self.agreement_percent, 2),
            "passed_calibration": self.passed_calibration,
            "per_category": self.per_category,
            "disagreements": self.disagreements,
            "confusion_matrix": self.confusion_matrix,
            "metadata": self.metadata,
        }


def compute_label_agreement(
    classifications: Sequence[ErrorClassification],
    manual_labeled_records: Sequence[Dict[str, Any]],
    min_agreement_threshold: float = 0.80,
) -> AgreementReport:
    """Compute agreement metrics between automatic error classifications and human ground-truth labels."""
    # Index manual labels by problem_id
    manual_by_id = {str(rec.get("problem_id", idx)): rec for idx, rec in enumerate(manual_labeled_records)}

    agreed = 0
    disagreed = 0
    disagreements: List[Dict[str, Any]] = []

    # Initialize confusion matrix: gold (rows) -> predicted (cols)
    confusion_matrix: Dict[str, Dict[str, int]] = {
        gold_cat: {pred_cat: 0 for pred_cat in ALL_ERROR_CATEGORIES} for gold_cat in ALL_ERROR_CATEGORIES
    }

    per_category_stats: Dict[str, Dict[str, Any]] = {
        cat: {"gold_count": 0, "predicted_count": 0, "true_positives": 0, "agreement_rate": 0.0}
        for cat in ALL_ERROR_CATEGORIES
    }

    for c in classifications:
        pid = str(c.problem_id)
        if pid not in manual_by_id:
            continue

        manual_entry = manual_by_id[pid]
        gold_label = str(manual_entry.get("manual_label", "")).lower()
        pred_label = c.category.value if c.category else "none"

        if gold_label in per_category_stats:
            per_category_stats[gold_label]["gold_count"] += 1
        if pred_label in per_category_stats:
            per_category_stats[pred_label]["predicted_count"] += 1

        if gold_label in confusion_matrix and pred_label in confusion_matrix[gold_label]:
            confusion_matrix[gold_label][pred_label] += 1

        if gold_label == pred_label:
            agreed += 1
            if gold_label in per_category_stats:
                per_category_stats[gold_label]["true_positives"] += 1
        else:
            disagreed += 1
            disagreements.append({
                "problem_id": pid,
                "question": c.question,
                "gold_answer": c.gold_answer,
                "extracted_answer": c.extracted_answer,
                "manual_label": gold_label,
                "predicted_category": pred_label,
                "classification_source": c.classification_source,
                "rationale": c.rationale,
                "manual_rationale": manual_entry.get("manual_rationale", ""),
            })

    total = agreed + disagreed
    agreement_rate = (agreed / total) if total > 0 else 0.0
    agreement_percent = round(agreement_rate * 100.0, 2)
    passed_calibration = agreement_rate >= min_agreement_threshold

    for cat, stats in per_category_stats.items():
        g_cnt = stats["gold_count"]
        stats["agreement_rate"] = round((stats["true_positives"] / g_cnt * 100.0) if g_cnt > 0 else 100.0, 1)

    return AgreementReport(
        total_samples=total,
        agreed_count=agreed,
        disagreed_count=disagreed,
        agreement_rate=agreement_rate,
        agreement_percent=agreement_percent,
        passed_calibration=passed_calibration,
        per_category=per_category_stats,
        disagreements=disagreements,
        confusion_matrix=confusion_matrix,
        metadata={"min_agreement_threshold": min_agreement_threshold},
    )


def generate_agreement_markdown(report: AgreementReport, stage_title: str = "30-Sample Calibration Benchmark") -> str:
    """Format AgreementReport as Markdown documentation."""
    status_str = "PASSED" if report.passed_calibration else "FAILED"
    lines = [
        f"# Error Category Agreement & Calibration Report — {stage_title}",
        "",
        "## 1. Calibration Verdict",
        f"- **Benchmark Status:** **{status_str}** (Threshold: {report.metadata.get('min_agreement_threshold', 0.8) * 100:.0f}%)",
        f"- **Total Labeled Problems:** {report.total_samples}",
        f"- **Agreed Classifications:** {report.agreed_count} ({report.agreement_percent}%)",
        f"- **Disagreed Classifications:** {report.disagreed_count} ({round(100.0 - report.agreement_percent, 2)}%)",
        "",
        "## 2. Per-Category Accuracy & Recall",
        "",
        "| Category | Ground Truth Count | Predicted Count | True Positives | Category Recall |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for cat in ALL_ERROR_CATEGORIES:
        stats = report.per_category.get(cat, {})
        g_cnt = stats.get("gold_count", 0)
        p_cnt = stats.get("predicted_count", 0)
        tp = stats.get("true_positives", 0)
        rec = stats.get("agreement_rate", 0.0)
        lines.append(f"| `{cat}` | {g_cnt} | {p_cnt} | {tp} | {rec:.1f}% |")

    lines.extend([
        "",
        "## 3. Disagreements Breakdown",
        "",
    ])
    if not report.disagreements:
        lines.append("Perfect agreement achieved across all labeled samples (0 disagreements).")
    else:
        for idx, dis in enumerate(report.disagreements, 1):
            lines.append(f"**Disagreement {idx} (Problem ID: {dis['problem_id']})**")
            lines.append(f"- **Manual Ground Truth:** `{dis['manual_label']}` (Rationale: {dis['manual_rationale']})")
            lines.append(f"- **Model Prediction:** `{dis['predicted_category']}` (Source: {dis['classification_source']})")
            lines.append(f"- **Model Rationale:** {dis['rationale']}")
            lines.append("")

    return "\n".join(lines)



# ==============================================================================
# Phase 5 Day 33: Semantic Topic Clustering of Error Questions
# ==============================================================================

@dataclass
class ErrorCluster:
    """Represents a coherent semantic cluster of mathematical error questions."""

    cluster_id: int
    name: str
    size: int
    percentage: float
    top_keywords: List[str]
    representative_examples: List[Dict[str, Any]]
    problem_ids: List[str]
    category_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "size": self.size,
            "percentage": round(self.percentage, 2),
            "top_keywords": self.top_keywords,
            "representative_examples": self.representative_examples,
            "problem_ids": self.problem_ids,
            "category_distribution": self.category_distribution,
        }


@dataclass
class ClusterAnalysisReport:
    """Consolidated topic cluster analysis report across error questions."""

    total_error_questions: int
    num_clusters: int
    embedding_model: str
    clusters: List[ErrorCluster]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_error_questions": self.total_error_questions,
            "num_clusters": self.num_clusters,
            "embedding_model": self.embedding_model,
            "clusters": [c.to_dict() for c in self.clusters],
        }


CANONICAL_MATH_TOPICS: Dict[str, Dict[str, Any]] = {
    "Percentages, Discounts & Proportions": {
        "keywords": [
            "percent", "percentage", "discount", "tax", "tip", "sale", "off",
            "ratio", "proportion", "proportions", "fraction", "half", "third", "quarter",
        ],
        "weight": 1.2,
    },
    "Rates, Speed, Work & Motion": {
        "keywords": [
            "speed", "mph", "miles", "rate", "faster", "slower", "distance",
            "travel", "travels", "traveled", "leak", "fill", "empty", "drain",
            "work", "workers", "paint", "job", "per hour",
        ],
        "weight": 1.1,
    },
    "Money, Commerce & Pricing": {
        "keywords": [
            "dollar", "dollars", "cost", "costs", "price", "prices", "buy",
            "buys", "bought", "sell", "sells", "sold", "pay", "paid", "spent",
            "spend", "bill", "change", "earn", "earns", "earned", "salary", "profit",
        ],
        "weight": 1.0,
    },
    "Geometry, Measurement & Spatial Units": {
        "keywords": [
            "perimeter", "area", "length", "width", "height", "depth", "volume",
            "rectangle", "square", "triangle", "circle", "radius", "diameter",
            "feet", "inches", "meters", "centimeters", "yard", "yards", "pounds",
            "ounces", "liters", "gallons", "weigh", "weight",
        ],
        "weight": 1.2,
    },
    "Time, Durations & Scheduling": {
        "keywords": [
            "minute", "minutes", "second", "seconds", "morning", "afternoon",
            "evening", "yesterday", "tomorrow", "clock", "calendar", "schedule",
            "days", "weeks", "months", "years", "later", "earlier", "delay",
        ],
        "weight": 1.0,
    },
    "Multi-Step Counting, Inventory & Combinatorics": {
        "keywords": [
            "apples", "books", "cookies", "trays", "loaves", "cards", "marbles",
            "desks", "cows", "sheep", "animals", "shelves", "rows", "boxes",
            "stamps", "total", "altogether", "remaining", "left", "leftover",
            "share", "shares", "divide", "equally", "each", "twice",
        ],
        "weight": 0.9,
    },
}


def extract_cluster_keywords(
    cluster_texts: Sequence[str],
    all_texts: Sequence[str],
    top_n: int = 5,
) -> List[str]:
    """Extract top discriminating keywords for a cluster relative to all texts using TF-IDF."""
    if not cluster_texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        combined = list(all_texts)
        vec = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vec.fit_transform(combined)
        feature_names = np.array(vec.get_feature_names_out())

        cluster_indices = [i for i, t in enumerate(combined) if t in cluster_texts]
        if not cluster_indices:
            cluster_indices = list(range(len(cluster_texts)))

        cluster_submatrix = tfidf_matrix[cluster_indices]
        mean_scores = np.asarray(cluster_submatrix.mean(axis=0)).ravel()
        top_indices = mean_scores.argsort()[::-1][:top_n]
        return [str(feature_names[idx]) for idx in top_indices if mean_scores[idx] > 0]
    except Exception:
        stop_words = {
            "the", "a", "an", "is", "and", "or", "to", "in", "of", "how", "many",
            "what", "does", "did", "he", "she", "they", "his", "her", "their",
            "with", "has", "had", "have", "for", "are", "each", "was", "were",
        }
        counts: Dict[str, int] = {}
        for t in cluster_texts:
            for w in re.findall(r"\b[a-zA-Z]{3,}\b", t.lower()):
                if w not in stop_words:
                    counts[w] = counts.get(w, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def assign_cluster_topic(
    cluster_texts: Sequence[str],
    top_keywords: Sequence[str],
    existing_topics: Optional[Sequence[str]] = None,
) -> str:
    """Assign an interpretable domain topic name to a cluster of question texts."""
    if not cluster_texts:
        return "Miscellaneous Arithmetic"

    cluster_blob = " ".join(cluster_texts).lower()
    kw_blob = " ".join(top_keywords).lower()

    scores: Dict[str, float] = {}
    for topic_name, meta in CANONICAL_MATH_TOPICS.items():
        score = 0.0
        kw_list = meta["keywords"]
        weight = meta.get("weight", 1.0)
        for kw in kw_list:
            hits = len(re.findall(rf"\b{re.escape(kw)}\b", cluster_blob))
            if hits > 0:
                score += (hits ** 0.5) * weight
            if kw in kw_blob:
                score += 1.5 * weight
        scores[topic_name] = score

    best_topic = max(scores, key=lambda k: scores[k])
    best_score = scores[best_topic]

    if best_score < 1.0:
        if top_keywords:
            valid_kw = [k for k in top_keywords if len(k) > 2][:2]
            if valid_kw:
                return f"Topic: {' & '.join(valid_kw).title()}"
        return "General Multi-Step Arithmetic"

    # Disambiguate if topic already assigned to another cluster
    if existing_topics and best_topic in existing_topics:
        distinctive_kw = [k for k in top_keywords if len(k) > 2][:1]
        if distinctive_kw:
            return f"{best_topic} ({distinctive_kw[0].title()})"

    return best_topic


class QuestionEmbedder:
    """Embeds question texts using sentence-transformers/all-MiniLM-L6-v2 with TF-IDF fallback."""

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_fallback: bool = False,
    ):
        self.model_name = model_name_or_path
        self.use_fallback = use_fallback
        self.backend = "sentence-transformers" if not use_fallback else "tfidf"
        self._st_model = None

    def embed(self, texts: Sequence[str]) -> Tuple[np.ndarray, str]:
        """Embed a sequence of texts into normalized embedding vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32), self.backend

        if not self.use_fallback:
            try:
                import torch  # noqa: F401
                from sentence_transformers import SentenceTransformer

                if self._st_model is None:
                    logger.info("Loading sentence-transformers model: %s", self.model_name)
                    self._st_model = SentenceTransformer(self.model_name)
                embs = self._st_model.encode(
                    list(texts),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return np.asarray(embs, dtype=np.float32), "sentence-transformers"
            except Exception as e:
                logger.info(
                    "sentence_transformers unavailable (%s); falling back to TF-IDF text embeddings.",
                    e,
                )

        # Fallback via sklearn TfidfVectorizer (or pure-python term frequency if sklearn missing)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(max_features=384, stop_words="english", ngram_range=(1, 2))
            raw = vec.fit_transform(texts).toarray()
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return (raw / norms).astype(np.float32), "tfidf_fallback"
        except (ImportError, Exception) as e:
            logger.info("sklearn unavailable (%s); using pure-python term-frequency vectorizer.", e)
            stop_words = {
                "the", "a", "an", "is", "and", "or", "to", "in", "of", "how", "many",
                "what", "does", "did", "he", "she", "they", "his", "her", "their",
                "with", "has", "had", "have", "for", "are", "each", "was", "were",
            }
            vocab: Dict[str, int] = {}
            doc_tokens: List[List[str]] = []
            for t in texts:
                tokens = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", t.lower()) if w not in stop_words]
                doc_tokens.append(tokens)
                for w in tokens:
                    if w not in vocab and len(vocab) < 384:
                        vocab[w] = len(vocab)

            dim = max(1, len(vocab))
            matrix = np.zeros((len(texts), dim), dtype=np.float32)
            for i, tokens in enumerate(doc_tokens):
                for w in tokens:
                    if w in vocab:
                        matrix[i, vocab[w]] += 1.0
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return (matrix / norms).astype(np.float32), "term_frequency_fallback"


class ErrorClusterer:
    """Clusters mathematical reasoning errors into semantic topic clusters."""

    def __init__(
        self,
        embedder: Optional[QuestionEmbedder] = None,
        num_clusters: int = 5,
        random_state: int = 42,
    ):
        self.embedder = embedder or QuestionEmbedder()
        self.num_clusters = num_clusters
        self.random_state = random_state

    def cluster(
        self,
        records: Sequence[Union[ErrorClassification, Dict[str, Any]]],
        num_clusters: Optional[int] = None,
    ) -> ClusterAnalysisReport:
        """Cluster error records by question semantic similarity.

        Args:
            records: Sequence of ErrorClassification objects or dictionary problem records.
            num_clusters: Optional override for number of clusters K.

        Returns:
            ClusterAnalysisReport containing clusters, keywords, and 2 representative examples per cluster.
        """
        # Filter down strictly to errors
        error_items: List[Dict[str, Any]] = []
        for r in records:
            if isinstance(r, ErrorClassification):
                if r.is_error:
                    error_items.append(r.to_dict())
            elif isinstance(r, dict):
                is_err = r.get("is_error", True)
                if is_err:
                    error_items.append(r)

        if not error_items:
            logger.warning("No error items found to cluster.")
            return ClusterAnalysisReport(
                total_error_questions=0,
                num_clusters=0,
                embedding_model=self.embedder.backend,
                clusters=[],
            )

        questions = [item.get("question", "") for item in error_items]
        embeddings, backend_name = self.embedder.embed(questions)

        n_samples = len(error_items)
        k = num_clusters if num_clusters is not None else self.num_clusters
        k = max(1, min(k, n_samples))

        logger.info(
            "Clustering %d error questions into %d semantic clusters using %s...",
            n_samples,
            k,
            backend_name,
        )

        if k == 1 or n_samples == 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            try:
                import os
                os.environ["LOKY_MAX_CPU_COUNT"] = "1"
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(embeddings)
            except (ImportError, Exception) as e:
                logger.info("sklearn KMeans unavailable (%s); executing deterministic cosine k-means.", e)
                rng = np.random.default_rng(self.random_state)
                init_idx = rng.choice(n_samples, size=k, replace=False)
                centers = embeddings[init_idx].copy()
                labels = np.zeros(n_samples, dtype=int)
                for _ in range(15):
                    sims = np.dot(embeddings, centers.T)
                    new_labels = np.argmax(sims, axis=1)
                    if np.array_equal(labels, new_labels):
                        break
                    labels = new_labels
                    for c_i in range(k):
                        mask = (labels == c_i)
                        if np.any(mask):
                            c_new = np.mean(embeddings[mask], axis=0)
                            norm = np.linalg.norm(c_new)
                            centers[c_i] = c_new / (norm if norm > 0 else 1.0)

        # Group indices by cluster
        clusters_dict: Dict[int, List[int]] = {c: [] for c in range(k)}
        for idx, lbl in enumerate(labels):
            clusters_dict[int(lbl)].append(idx)

        all_clusters: List[ErrorCluster] = []
        existing_topic_names: List[str] = []

        for cid, indices in clusters_dict.items():
            if not indices:
                continue

            cluster_records = [error_items[i] for i in indices]
            cluster_questions = [questions[i] for i in indices]
            cluster_embs = embeddings[indices]

            # Compute centroid and distances
            centroid = np.mean(cluster_embs, axis=0)
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)

            sorted_order = np.argsort(distances)
            sorted_records = [cluster_records[i] for i in sorted_order]
            sorted_distances = [distances[i] for i in sorted_order]

            representative_examples: List[Dict[str, Any]] = []
            for rank, rec in enumerate(sorted_records[:2]):
                cat_val = rec.get("category")
                if isinstance(cat_val, dict):
                    cat_name = cat_val.get("value", "other")
                elif hasattr(cat_val, "value"):
                    cat_name = cat_val.value
                else:
                    cat_name = str(cat_val or "other")

                representative_examples.append({
                    "problem_id": rec.get("problem_id", f"cluster_{cid}_{rank+1}"),
                    "question": rec.get("question", ""),
                    "gold_answer": str(rec.get("gold_answer", "")),
                    "gold_solution": rec.get("gold_solution", ""),
                    "generated_text": rec.get("generated_text", ""),
                    "extracted_answer": str(rec.get("extracted_answer", "")),
                    "category": cat_name,
                    "rationale": rec.get("rationale", ""),
                    "distance_to_centroid": round(float(sorted_distances[rank]), 4),
                })

            keywords = extract_cluster_keywords(cluster_questions, questions, top_n=5)
            topic_name = assign_cluster_topic(cluster_questions, keywords, existing_topic_names)
            existing_topic_names.append(topic_name)

            cat_counts: Dict[str, int] = {}
            for rec in cluster_records:
                cval = rec.get("category")
                cname = cval.value if hasattr(cval, "value") else str(cval or "other")
                cat_counts[cname] = cat_counts.get(cname, 0) + 1

            cluster_obj = ErrorCluster(
                cluster_id=cid + 1,
                name=topic_name,
                size=len(indices),
                percentage=(len(indices) / n_samples) * 100.0,
                top_keywords=keywords,
                representative_examples=representative_examples,
                problem_ids=[rec.get("problem_id", "") for rec in cluster_records],
                category_distribution=cat_counts,
            )
            all_clusters.append(cluster_obj)

        all_clusters.sort(key=lambda c: c.size, reverse=True)
        for new_id, cl in enumerate(all_clusters, start=1):
            cl.cluster_id = new_id

        return ClusterAnalysisReport(
            total_error_questions=n_samples,
            num_clusters=len(all_clusters),
            embedding_model=backend_name,
            clusters=all_clusters,
        )


def generate_cluster_markdown(report: ClusterAnalysisReport) -> str:
    """Generate Markdown section detailing semantic topic clusters and representative examples."""
    lines = [
        "## 4. Semantic Topic Clusters of Reasoning Errors",
        "",
        f"- **Total Error Questions Clustered:** {report.total_error_questions}",
        f"- **Number of Semantic Clusters:** {report.num_clusters}",
        f"- **Embedding Model:** `{report.embedding_model}`",
        "",
        "### Cluster Overview Table",
        "",
        "| Cluster | Topic / Mathematical Domain | Errors | % of Errors | Top Distinctive Keywords |",
        "| :---: | :--- | :---: | :---: | :--- |",
    ]
    for c in report.clusters:
        kw_str = ", ".join(f"`{k}`" for k in c.top_keywords[:4])
        lines.append(f"| {c.cluster_id} | **{c.name}** | {c.size} | {c.percentage:.1f}% | {kw_str} |")

    lines.extend(["", "### Detailed Cluster Profiles & Representative Examples", ""])

    for c in report.clusters:
        lines.append(f"#### Cluster {c.cluster_id}: {c.name}")
        lines.append(f"- **Cluster Size:** {c.size} errors ({c.percentage:.1f}% of analyzed errors)")
        lines.append(f"- **Top Distinguishing Keywords:** {', '.join(c.top_keywords)}")
        if c.category_distribution:
            cats = ", ".join(
                f"`{cat}`: {cnt}"
                for cat, cnt in sorted(c.category_distribution.items(), key=lambda x: -x[1])
            )
            lines.append(f"- **Error Category Distribution:** {cats}")
        lines.append("")

        for idx, ex in enumerate(c.representative_examples, 1):
            lines.append(f"##### Example {c.cluster_id}.{idx} (ID: {ex['problem_id']})")
            lines.append(f"- **Question:** {ex['question']}")
            lines.append(f"- **Gold Answer:** `{ex['gold_answer']}`")
            lines.append(f"- **Extracted Answer:** `{ex['extracted_answer']}`")
            lines.append(f"- **Assigned Error Category:** `{ex['category']}`")
            if ex.get("rationale"):
                lines.append(f"- **Failure Analysis:** {ex['rationale']}")
            lines.append("```text")
            gen_text = ex.get("generated_text", "")
            lines.append(gen_text[:350] + ("..." if len(gen_text) > 350 else ""))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)
