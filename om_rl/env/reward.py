"""Reward computation for the Opus Magnum RL environment.

Two reward modes (toggled via config flags):

1. Basic mode (default before structure scoring):
   - Correct: positive, scaled by token efficiency
   - Parseable but wrong: -0.1 (+ optional progress score)
   - Unparseable: -0.5

2. Structure scoring mode (use_structure_scoring=True):
   - Maps the structure score (0.0-1.0) to a reward range
   - Provides smooth gradient from "total garbage" to "nearly correct"
   - This is the recommended mode for training — without it, almost all
     failed attempts get the same reward and GRPO has no signal.

Both modes can be active simultaneously. Structure scoring replaces
the coarse parseable/unparseable distinction with a fine-grained score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    token_budget: int = 4096
    correct_base: float = 1.0
    budget_exceeded: float = 0.0

    # Structure scoring (recommended — provides reward differentiation)
    use_structure_scoring: bool = True
    structure_reward_min: float = -0.5  # Reward for structure score 0.0
    structure_reward_max: float = 0.0   # Reward for structure score ~0.9

    # Legacy intermediate rewards (from complexity evaluator trace)
    use_intermediate_rewards: bool = False
    intermediate_weight: float = 0.3

    # Legacy fixed rewards (used when structure scoring is off)
    parseable_wrong: float = -0.1
    unparseable: float = -0.5


def compute_reward(
    verified: bool,
    tokens_used: int,
    config: RewardConfig | None = None,
    structure_score: float | None = None,
    progress_score: float = 0.0,
    parseable: bool = False,
) -> float:
    """Compute reward for a solution attempt.

    Args:
        verified: Whether omsim verified the solution as correct.
        tokens_used: Total tokens used so far in this episode.
        config: Reward configuration.
        structure_score: Score from structure_scorer (0.0-1.0).
            Used when config.use_structure_scoring is True.
        progress_score: Score from complexity evaluator trace (0-1).
            Used when config.use_intermediate_rewards is True.
        parseable: Whether the solution text could be parsed.
            Used as fallback when structure scoring is off.

    Returns:
        Reward value.
    """
    if config is None:
        config = RewardConfig()

    if tokens_used > config.token_budget:
        return config.budget_exceeded

    if verified:
        efficiency = 1.0 - (tokens_used / config.token_budget)
        return max(0.01, config.correct_base * efficiency)

    # Structure scoring: smooth reward based on how structurally correct
    if config.use_structure_scoring and structure_score is not None:
        # Linear map: score 0.0 -> reward_min, score 1.0 -> reward_max
        reward = (
            config.structure_reward_min
            + (config.structure_reward_max - config.structure_reward_min) * structure_score
        )
        return reward

    # Fallback: coarse parseable/unparseable distinction
    if parseable:
        base = config.parseable_wrong
        if config.use_intermediate_rewards and progress_score > 0:
            base += config.intermediate_weight * progress_score
        return base

    return config.unparseable
