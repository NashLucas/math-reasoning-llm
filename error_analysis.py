"""Root entrypoint and module re-export for error analysis (Phase 5 Day 31)."""

import sys

from src.evaluation.error_analysis import (
    ALL_ERROR_CATEGORIES,
    RULE_BASED_CATEGORIES,
    SEMANTIC_CATEGORIES,
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    check_parse_failure,
    check_truncation,
    classify_semantic_heuristic,
    compute_error_distribution,
    format_error_classification_prompt,
    generate_error_markdown_report,
    parse_llm_classification,
)

__all__ = [
    "ALL_ERROR_CATEGORIES",
    "RULE_BASED_CATEGORIES",
    "SEMANTIC_CATEGORIES",
    "ErrorCategory",
    "ErrorClassification",
    "ErrorClassifier",
    "check_parse_failure",
    "check_truncation",
    "classify_semantic_heuristic",
    "compute_error_distribution",
    "format_error_classification_prompt",
    "generate_error_markdown_report",
    "parse_llm_classification",
]

if __name__ == "__main__":
    from scripts.error_analysis import main

    sys.exit(main())
