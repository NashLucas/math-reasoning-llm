"""Evaluation modules and tools for GSM8K reasoning models."""

from src.evaluation.category_shift import (
    CategoryShiftMetric,
    CategoryShiftSummary,
    ProblemTransitionRecord,
    compute_category_shift,
    generate_category_shift_table,
    run_category_shift_analysis,
)
from src.evaluation.evaluator import MathEvaluator, TestSetWallError
from src.evaluation.phase5_verifier import (
    GateCriterionStatus,
    Phase5GateSummary,
    generate_phase5_wrapup_report,
    run_phase5_wrapup,
    save_phase5_summary_json,
    verify_all_phase5_gates,
)
from src.evaluation.round_comparator import (
    ComparisonDecision,
    ProblemComparisonRecord,
    RoundEvalMetrics,
    compare_problem_level_results,
    compute_comparison_decision,
    evaluate_round,
    resolve_round_adapter,
    run_round_comparison,
)

__all__ = [
    "MathEvaluator",
    "TestSetWallError",
    "RoundEvalMetrics",
    "ComparisonDecision",
    "ProblemComparisonRecord",
    "compare_problem_level_results",
    "compute_comparison_decision",
    "evaluate_round",
    "resolve_round_adapter",
    "run_round_comparison",
    "CategoryShiftMetric",
    "CategoryShiftSummary",
    "ProblemTransitionRecord",
    "compute_category_shift",
    "generate_category_shift_table",
    "run_category_shift_analysis",
    "GateCriterionStatus",
    "Phase5GateSummary",
    "verify_all_phase5_gates",
    "generate_phase5_wrapup_report",
    "save_phase5_summary_json",
    "run_phase5_wrapup",
]
