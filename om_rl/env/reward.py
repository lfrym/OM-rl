"""Reward computation for the Opus Magnum RL environment.

Reward scheme:
- Correct solution: positive reward scaled by token efficiency
- Parseable but incorrect: small negative, plus optional partial credit
  from complexity evaluator (intermediate rewards)
- Unparseable: larger negative
- Token budget exceeded: zero (episode terminated)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    token_budget: int = 4096
    correct_base: float = 1.0
    parseable_wrong: float = -0.1
    unparseable: float = -0.5
    budget_exceeded: float = 0.0

    # Intermediate reward settings
    use_intermediate_rewards: bool = False
    intermediate_weight: float = 0.3  # How much to weight progress score


def compute_reward(
    verified: bool,
    parseable: bool,
    tokens_used: int,
    config: RewardConfig | None = None,
    progress_score: float = 0.0,
) -> float:
    """Compute reward for a solution attempt.

    Args:
        verified: Whether omsim verified the solution as correct.
        parseable: Whether the solution text could be parsed at all.
        tokens_used: Total tokens used so far in this episode.
        config: Reward configuration. Uses defaults if None.
        progress_score: Partial progress score (0-1) from complexity evaluator.
            Only used when config.use_intermediate_rewards is True.

    Returns:
        Reward value.
    """
    if config is None:
        config = RewardConfig()

    if tokens_used > config.token_budget:
        return config.budget_exceeded

    if verified:
        # Reward decreases linearly with token usage
        efficiency = 1.0 - (tokens_used / config.token_budget)
        return max(0.01, config.correct_base * efficiency)  # minimum small positive

    if parseable:
        base = config.parseable_wrong
        if config.use_intermediate_rewards and progress_score > 0:
            # Blend: base negative + partial credit for progress
            # e.g., with weight=0.3 and progress=0.5: -0.1 + 0.3*0.5 = +0.05
            base += config.intermediate_weight * progress_score
        return base

    return config.unparseable
