"""Evaluation harness for trained models.

Evaluates a model on a set of puzzles and reports solve rate,
token usage, and solution metrics. Supports multi-turn evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vendor.opus_magnum.models import Puzzle

from om_rl.env.reward import RewardConfig
from .rollout import collect_rollouts, RolloutBatch

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    num_puzzles: int
    num_solved: int
    solve_rate: float
    mean_reward: float
    mean_tokens: float
    mean_attempts: float
    per_puzzle: list[dict[str, Any]]

    def summary(self) -> str:
        lines = [
            f"Evaluation Results ({self.num_puzzles} puzzles):",
            f"  Solve rate: {self.solve_rate:.2%} ({self.num_solved}/{self.num_puzzles})",
            f"  Mean reward: {self.mean_reward:.3f}",
            f"  Mean tokens: {self.mean_tokens:.0f}",
            f"  Mean attempts: {self.mean_attempts:.1f}",
        ]
        if self.num_solved > 0:
            solved = [p for p in self.per_puzzle if p.get("verified")]
            solved_with_metrics = [p for p in solved if p.get("metrics")]
            if solved_with_metrics:
                mean_cost = sum(p["metrics"]["cost"] for p in solved_with_metrics) / len(solved_with_metrics)
                mean_cycles = sum(p["metrics"]["cycles"] for p in solved_with_metrics) / len(solved_with_metrics)
                lines.append(f"  Solved — mean cost: {mean_cost:.0f}, mean cycles: {mean_cycles:.0f}")
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({
                "num_puzzles": self.num_puzzles,
                "num_solved": self.num_solved,
                "solve_rate": self.solve_rate,
                "mean_reward": self.mean_reward,
                "mean_tokens": self.mean_tokens,
                "mean_attempts": self.mean_attempts,
                "per_puzzle": self.per_puzzle,
            }, f, indent=2)


def evaluate(
    puzzles: list[Puzzle],
    generate_fn,
    reward_config: RewardConfig | None = None,
    cycle_limit: int = 100_000,
    max_attempts: int = 3,
) -> EvalResult:
    """Evaluate a model on a set of puzzles.

    Args:
        puzzles: Puzzles to evaluate on.
        generate_fn: Callable(prompt: str) -> tuple[str, int]
        reward_config: Reward configuration.
        cycle_limit: omsim cycle limit.
        max_attempts: Max submissions per puzzle (multi-turn).
    """
    batch = collect_rollouts(
        puzzles, generate_fn, reward_config, cycle_limit,
        max_attempts=max_attempts,
    )

    per_puzzle = []
    for ep in batch.results:
        per_puzzle.append({
            "puzzle_name": ep.puzzle_name,
            "verified": ep.verified,
            "reward": ep.final_reward,
            "tokens_used": ep.total_tokens,
            "num_attempts": ep.num_attempts,
            "metrics": ep.metrics,
        })

    stats = batch.stats()
    return EvalResult(
        num_puzzles=len(puzzles),
        num_solved=batch.num_verified,
        solve_rate=batch.solve_rate,
        mean_reward=batch.mean_reward,
        mean_tokens=stats["mean_tokens"],
        mean_attempts=stats["mean_attempts"],
        per_puzzle=per_puzzle,
    )
