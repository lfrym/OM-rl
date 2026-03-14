"""RL environment for Opus Magnum puzzle solving.

Wraps puzzle generation, solution parsing, and omsim verification
into a simple reset/step interface suitable for RL training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vendor.opus_magnum.models import Puzzle
from vendor.opus_magnum.text_format import parse_text_solution
from vendor.opus_magnum.solution_writer import write_solution
from vendor.opus_magnum.verifier import Metrics, Verifier, VerificationError

from .observation import format_initial_observation, format_feedback_observation
from .reward import RewardConfig, compute_reward

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of an environment step."""

    observation: str  # Next observation text for the model
    reward: float
    done: bool  # Episode is over
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Configuration for the RL environment."""

    max_attempts: int = 10  # Max solution submissions per episode
    cycle_limit: int = 100_000  # omsim cycle limit
    reward_config: RewardConfig = field(default_factory=RewardConfig)


class OpusMagnumEnv:
    """RL environment for Opus Magnum puzzle solving."""

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        self._puzzle: Puzzle | None = None
        self._attempt: int = 0
        self._total_tokens: int = 0
        self._solved: bool = False

    @property
    def puzzle(self) -> Puzzle | None:
        return self._puzzle

    @property
    def attempt_number(self) -> int:
        return self._attempt

    @property
    def is_solved(self) -> bool:
        return self._solved

    def reset(self, puzzle: Puzzle) -> str:
        """Start a new episode with the given puzzle."""
        self._puzzle = puzzle
        self._attempt = 0
        self._total_tokens = 0
        self._solved = False
        return format_initial_observation(puzzle)

    def step(self, solution_text: str, tokens_used: int) -> StepResult:
        """Submit a solution attempt and get feedback."""
        if self._puzzle is None:
            raise RuntimeError("Call reset() before step()")

        self._attempt += 1
        self._total_tokens = tokens_used
        rcfg = self.config.reward_config

        info: dict[str, Any] = {
            "attempt": self._attempt,
            "tokens_used": tokens_used,
            "puzzle_name": self._puzzle.name,
        }

        # Check token budget
        if tokens_used > rcfg.token_budget:
            return StepResult(
                observation="Token budget exceeded. Episode terminated.",
                reward=compute_reward(False, tokens_used, rcfg),
                done=True,
                info={**info, "termination": "budget_exceeded"},
            )

        # Compute structure score (always, for logging — only affects reward if enabled)
        from om_rl.complexity.structure_scorer import score_solution_structure
        omsim_error: str | None = None
        omsim_error_cycle: int = -1
        omsim_verified = False

        # Try to parse the solution
        try:
            solution = parse_text_solution(solution_text, self._puzzle.name)
            parseable = True
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            omsim_error = f"Could not parse solution: {e}"

            struct = score_solution_structure(
                solution_text, self._puzzle,
                omsim_error=omsim_error,
            )
            reward = compute_reward(
                False, tokens_used, rcfg,
                structure_score=struct.score,
                parseable=False,
            )
            done = self._attempt >= self.config.max_attempts
            obs = format_feedback_observation(
                self._puzzle, omsim_error,
                self._attempt, self.config.max_attempts,
            )
            return StepResult(
                observation=obs, reward=reward, done=done,
                info={
                    **info, "error": "parse_error",
                    "error_message": str(e),
                    "structure_score": struct.score,
                    "structure_level": struct.level,
                    "structure_desc": struct.description,
                },
            )

        # Try to verify with omsim
        try:
            sol_bytes = write_solution(solution)
            metrics = _verify_solution(
                self._puzzle.raw_bytes, sol_bytes, self.config.cycle_limit
            )
            self._solved = True
            omsim_verified = True

            struct = score_solution_structure(
                solution_text, self._puzzle, omsim_verified=True,
            )
            reward = compute_reward(True, tokens_used, rcfg, structure_score=1.0)
            return StepResult(
                observation=f"Solution verified! Metrics: cost={metrics.cost}, "
                f"cycles={metrics.cycles}, area={metrics.area}, "
                f"instructions={metrics.instructions}",
                reward=reward, done=True,
                info={
                    **info, "verified": True,
                    "structure_score": 1.0,
                    "structure_level": 10,
                    "metrics": {
                        "cost": metrics.cost, "cycles": metrics.cycles,
                        "area": metrics.area, "instructions": metrics.instructions,
                    },
                },
            )
        except VerificationError as e:
            omsim_error = str(e)
            omsim_error_cycle = e.cycle
            location = e.location

            detail_parts = [omsim_error]
            if omsim_error_cycle >= 0:
                detail_parts.append(f"Error at cycle {omsim_error_cycle}")
            if location:
                detail_parts.append(f"Error at hex ({location[0]}, {location[1]})")

            struct = score_solution_structure(
                solution_text, self._puzzle,
                omsim_error=omsim_error,
                omsim_error_cycle=omsim_error_cycle,
            )

            # Also compute legacy progress score if enabled
            progress = 0.0
            if rcfg.use_intermediate_rewards:
                try:
                    from om_rl.complexity.evaluator import evaluate_progress
                    progress_result = evaluate_progress(
                        self._puzzle, sol_bytes, max_trace_cycles=50
                    )
                    progress = progress_result.score
                except Exception:
                    pass

            reward = compute_reward(
                False, tokens_used, rcfg,
                structure_score=struct.score,
                progress_score=progress,
                parseable=True,
            )
            done = self._attempt >= self.config.max_attempts
            obs = format_feedback_observation(
                self._puzzle, "\n".join(detail_parts),
                self._attempt, self.config.max_attempts,
            )
            return StepResult(
                observation=obs, reward=reward, done=done,
                info={
                    **info, "error": "verification_error",
                    "error_message": omsim_error,
                    "error_cycle": omsim_error_cycle,
                    "error_location": location,
                    "structure_score": struct.score,
                    "structure_level": struct.level,
                    "structure_desc": struct.description,
                    "progress_score": progress,
                },
            )
        except Exception as e:
            logger.error(f"Unexpected verification error: {e}")
            struct = score_solution_structure(
                solution_text, self._puzzle,
                omsim_error=str(e),
            )
            reward = compute_reward(
                False, tokens_used, rcfg,
                structure_score=struct.score,
                parseable=True,
            )
            done = self._attempt >= self.config.max_attempts
            obs = format_feedback_observation(
                self._puzzle, f"Internal error during verification: {e}",
                self._attempt, self.config.max_attempts,
            )
            return StepResult(
                observation=obs, reward=reward, done=done,
                info={
                    **info, "error": "internal_error",
                    "error_message": str(e),
                    "structure_score": struct.score,
                    "structure_level": struct.level,
                },
            )


def _verify_solution(
    puzzle_bytes: bytes, solution_bytes: bytes, cycle_limit: int
) -> Metrics:
    """Verify a solution and return metrics."""
    with Verifier(puzzle_bytes, solution_bytes, cycle_limit) as v:
        return v.evaluate()
