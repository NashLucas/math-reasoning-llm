"""Re-export self-consistency majority voting from reasoning module for convenience."""

from src.reasoning.self_consistency import (
    is_valid_candidate,
    majority_vote,
    majority_vote_from_completions,
    majority_vote_with_details,
)

__all__ = [
    "is_valid_candidate",
    "majority_vote",
    "majority_vote_from_completions",
    "majority_vote_with_details",
]
