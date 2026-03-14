"""Rollout collection: generate solution attempts and compute rewards.

Supports two modes:
- Single-turn: model generates one solution, gets one reward.
- Multi-turn: model generates a solution, sees error feedback + progress,
  generates again, etc. The full trajectory is the "completion" for GRPO.

In multi-turn mode, the model gets to "touch reality" by testing solutions
against omsim and iterating based on feedback. This gives much denser
learning signal than single-turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vendor.opus_magnum.models import Puzzle

from om_rl.env.environment import OpusMagnumEnv, EnvironmentConfig, StepResult
from om_rl.env.observation import format_initial_observation
from om_rl.env.reward import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single turn within an episode: model output + environment response."""

    generation: str  # What the model generated
    tokens: int  # Tokens used for this generation
    observation: str  # Feedback from environment
    reward: float  # Reward for this turn
    verified: bool  # Whether this turn solved the puzzle
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result of a full episode (one or more turns) on a single puzzle."""

    puzzle_name: str
    prompt: str  # Initial observation (the "prompt" for GRPO)
    turns: list[Turn]  # All turns in the episode

    @property
    def trajectory(self) -> str:
        """The full trajectory text (all generations + feedback concatenated).

        This is the "completion" for GRPO — the entire multi-turn interaction.
        """
        parts: list[str] = []
        for turn in self.turns:
            parts.append(turn.generation)
            if not turn.verified:
                parts.append(f"\n\n{turn.observation}\n\n")
        return "".join(parts)

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens for t in self.turns)

    @property
    def final_reward(self) -> float:
        """Reward for the whole episode: best turn's reward.

        Using best (not last) because we want to reward the model for
        ever producing a correct solution, even if it kept going after.
        """
        if not self.turns:
            return -0.5
        return max(t.reward for t in self.turns)

    @property
    def verified(self) -> bool:
        return any(t.verified for t in self.turns)

    @property
    def num_attempts(self) -> int:
        return len(self.turns)

    @property
    def metrics(self) -> dict[str, int] | None:
        """Metrics from the first successful verification, if any."""
        for t in self.turns:
            if t.verified and "metrics" in t.info:
                return t.info["metrics"]
        return None


@dataclass
class RolloutBatch:
    """A batch of episode results for training."""

    results: list[EpisodeResult]

    @property
    def mean_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.final_reward for r in self.results) / len(self.results)

    @property
    def solve_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.verified) / len(self.results)

    @property
    def num_verified(self) -> int:
        return sum(1 for r in self.results if r.verified)

    def stats(self) -> dict[str, Any]:
        return {
            "num_episodes": len(self.results),
            "mean_reward": self.mean_reward,
            "solve_rate": self.solve_rate,
            "num_verified": self.num_verified,
            "mean_tokens": (
                sum(r.total_tokens for r in self.results) / len(self.results)
                if self.results
                else 0
            ),
            "mean_attempts": (
                sum(r.num_attempts for r in self.results) / len(self.results)
                if self.results
                else 0
            ),
        }


def collect_rollouts(
    puzzles: list[Puzzle],
    generate_fn,
    reward_config: RewardConfig | None = None,
    cycle_limit: int = 100_000,
    max_attempts: int = 1,
) -> RolloutBatch:
    """Collect rollouts for a batch of puzzles.

    Args:
        puzzles: Puzzles to solve.
        generate_fn: Callable(prompt: str) -> tuple[str, int] that generates
            a solution text and returns (solution_text, tokens_used).
            In multi-turn mode, the prompt grows with each turn (includes
            prior attempts and feedback).
        reward_config: Reward configuration.
        cycle_limit: omsim cycle limit.
        max_attempts: Max solution submissions per episode.
            1 = single-turn (original behavior).
            >1 = multi-turn (model iterates with feedback).

    Returns:
        RolloutBatch with all episode results.
    """
    env_config = EnvironmentConfig(
        max_attempts=max_attempts,
        cycle_limit=cycle_limit,
        reward_config=reward_config or RewardConfig(),
    )
    env = OpusMagnumEnv(env_config)
    results: list[EpisodeResult] = []

    for puzzle in puzzles:
        initial_obs = env.reset(puzzle)
        episode = EpisodeResult(
            puzzle_name=puzzle.name,
            prompt=initial_obs,
            turns=[],
        )

        # Build up context across turns
        context = initial_obs
        total_tokens = 0

        for attempt in range(max_attempts):
            import time as _time
            gen_start = _time.monotonic()

            try:
                solution_text, tokens_used = generate_fn(context)
            except Exception as e:
                logger.error(f"Generation failed for {puzzle.name} attempt {attempt+1}: {e}")
                episode.turns.append(Turn(
                    generation="",
                    tokens=0,
                    observation=f"Generation error: {e}",
                    reward=-0.5,
                    verified=False,
                    info={"error": "generation_error"},
                ))
                break

            gen_elapsed = _time.monotonic() - gen_start
            tok_per_sec = tokens_used / gen_elapsed if gen_elapsed > 0 else 0
            total_tokens += tokens_used

            step_result = env.step(solution_text, total_tokens)
            verified = step_result.info.get("verified", False)
            error_msg = step_result.info.get("error_message", "")
            progress = step_result.info.get("progress_score", "")

            # Log generation details including a preview of what the model produced
            preview = solution_text.replace('\n', ' ')[:120]
            logger.info(
                f"  {puzzle.name} attempt {attempt+1}/{max_attempts}: "
                f"{tokens_used} tok in {gen_elapsed:.1f}s ({tok_per_sec:.0f} tok/s) "
                f"{'SOLVED' if verified else 'FAILED'}"
                f"{f' err={error_msg[:60]}' if error_msg else ''}"
                f"{f' progress={progress:.2f}' if isinstance(progress, float) and progress > 0 else ''}"
            )
            logger.info(f"    preview: {preview}...")

            episode.turns.append(Turn(
                generation=solution_text,
                tokens=tokens_used,
                observation=step_result.observation,
                reward=step_result.reward,
                verified=verified,
                info=step_result.info,
            ))

            if step_result.done:
                break

            # Append feedback to context for next turn
            context = context + "\n\n" + solution_text + "\n\n" + step_result.observation + "\n\n"

        results.append(episode)

    return RolloutBatch(results=results)
