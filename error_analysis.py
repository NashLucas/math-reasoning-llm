"""Root entrypoint and module re-export for error analysis (Phase 5)."""

import sys

from src.evaluation.error_analysis import (
    ALL_ERROR_CATEGORIES,
    CANONICAL_MATH_TOPICS,
    RULE_BASED_CATEGORIES,
    SEMANTIC_CATEGORIES,
    AgreementReport,
    ClusterAnalysisReport,
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    ErrorCluster,
    ErrorClusterer,
    QuestionEmbedder,
    assign_cluster_topic,
    check_parse_failure,
    check_truncation,
    classify_semantic_heuristic,
    compute_error_distribution,
    compute_label_agreement,
    extract_cluster_keywords,
    format_error_classification_prompt,
    generate_agreement_markdown,
    generate_cluster_markdown,
    generate_error_markdown_report,
    parse_llm_classification,
)

__all__ = [
    "ALL_ERROR_CATEGORIES",
    "CANONICAL_MATH_TOPICS",
    "RULE_BASED_CATEGORIES",
    "SEMANTIC_CATEGORIES",
    "AgreementReport",
    "ClusterAnalysisReport",
    "ErrorCategory",
    "ErrorClassification",
    "ErrorClassifier",
    "ErrorCluster",
    "ErrorClusterer",
    "QuestionEmbedder",
    "assign_cluster_topic",
    "check_parse_failure",
    "check_truncation",
    "classify_semantic_heuristic",
    "compute_error_distribution",
    "compute_label_agreement",
    "extract_cluster_keywords",
    "format_error_classification_prompt",
    "generate_agreement_markdown",
    "generate_cluster_markdown",
    "generate_error_markdown_report",
    "parse_llm_classification",
]

if __name__ == "__main__":
    from scripts.error_analysis import main

    sys.exit(main())
