"""Multi-level logging for OM-rl training.

Verbosity levels:
  0 (QUIET):   Step summaries only (loss, reward, solve rate)
  1 (NORMAL):  + per-episode summaries (attempts, tokens, solved/failed)
  2 (VERBOSE): + per-attempt details (tok/s, error type, progress score, output preview)
  3 (TRACE):   + full model outputs, full feedback text, full trajectories

Usage:
    from om_rl.utils.logging import TrainingLogger
    tlog = TrainingLogger(verbosity=2, log_dir="outputs/logs")
    tlog.step_start(step=1, puzzles=["PUZZLE-A"])
    tlog.attempt(puzzle="PUZZLE-A", attempt=1, max_attempts=3,
                 generation="ARM arm1 ...", tokens=200, elapsed=12.3,
                 verified=False, error="collision", progress=0.3,
                 feedback="Attempt 1/3 failed...")
    tlog.episode_end(puzzle="PUZZLE-A", episode=1, attempts=3,
                     total_tokens=600, reward=-0.07, verified=False)
    tlog.step_end(step=1, loss=0.5, mean_reward=-0.07, solve_rate=0.0,
                  mean_attempts=3.0, mean_tokens=600, level=1, elapsed=45.2)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("om_rl.training")

QUIET = 0
NORMAL = 1
VERBOSE = 2
TRACE = 3


class TrainingLogger:
    """Structured multi-level logger for training runs."""

    def __init__(
        self,
        verbosity: int = NORMAL,
        log_dir: str | Path | None = None,
    ):
        self.verbosity = verbosity
        self.log_dir = Path(log_dir) if log_dir else None

        # Separate file for full traces (verbosity 3 can produce a LOT of output)
        self._trace_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._trace_file = open(self.log_dir / "trace.jsonl", "a")

    def close(self) -> None:
        if self._trace_file:
            self._trace_file.close()

    # ── Step-level logging (verbosity >= 0) ──────────────────────────

    def step_start(self, step: int, puzzles: list[str], num_completions: int) -> None:
        if self.verbosity >= NORMAL:
            logger.info(
                f"Step {step}: generating {num_completions} episodes "
                f"for {len(puzzles)} puzzle(s): {puzzles}"
            )
        elif self.verbosity >= QUIET:
            logger.info(f"Step {step}: {len(puzzles)} puzzle(s)...")

    def step_end(
        self,
        step: int,
        loss: float,
        mean_reward: float,
        solve_rate: float,
        mean_attempts: float,
        mean_tokens: float,
        level: int,
        elapsed: float,
        num_verified: int = 0,
        total_episodes: int = 0,
    ) -> None:
        logger.info(
            f"Step {step} DONE: loss={loss:.4f} "
            f"reward={mean_reward:.3f} "
            f"solve={solve_rate:.2%} ({num_verified}/{total_episodes}) "
            f"attempts={mean_attempts:.1f} "
            f"tokens={mean_tokens:.0f} "
            f"level={level} "
            f"time={elapsed:.1f}s"
        )

        if self._trace_file:
            self._write_trace({
                "event": "step_end",
                "step": step,
                "loss": loss,
                "mean_reward": mean_reward,
                "solve_rate": solve_rate,
                "mean_attempts": mean_attempts,
                "mean_tokens": mean_tokens,
                "level": level,
                "elapsed": elapsed,
            })

    # ── Episode-level logging (verbosity >= 1) ───────────────────────

    def episode_start(self, puzzle: str, episode: int, num_completions: int) -> None:
        if self.verbosity >= NORMAL:
            logger.info(f"  Episode {episode}/{num_completions}: {puzzle}")

    def episode_end(
        self,
        puzzle: str,
        episode: int,
        attempts: int,
        total_tokens: int,
        reward: float,
        verified: bool,
    ) -> None:
        if self.verbosity >= NORMAL:
            status = "SOLVED" if verified else "FAILED"
            logger.info(
                f"    -> {status} | {attempts} attempts | "
                f"{total_tokens} tokens | reward={reward:.3f}"
            )

    # ── Attempt-level logging (verbosity >= 2) ───────────────────────

    def attempt(
        self,
        puzzle: str,
        attempt: int,
        max_attempts: int,
        generation: str,
        tokens: int,
        elapsed: float,
        verified: bool,
        error: str | None = None,
        progress: float | None = None,
        feedback: str | None = None,
        structure_score: float | None = None,
        structure_level: int | None = None,
        structure_desc: str | None = None,
    ) -> None:
        tok_per_sec = tokens / elapsed if elapsed > 0 else 0

        if self.verbosity >= VERBOSE:
            preview = generation.replace('\n', '\\n')[:150]
            status = "SOLVED" if verified else "FAILED"
            parts = [
                f"    Attempt {attempt}/{max_attempts}: "
                f"{tokens} tok in {elapsed:.1f}s ({tok_per_sec:.0f} tok/s) {status}",
            ]
            if structure_score is not None:
                parts.append(f"      Structure: L{structure_level} score={structure_score:.2f} ({structure_desc})")
            if error:
                parts.append(f"      Error: {error[:100]}")
            if progress is not None and progress > 0:
                parts.append(f"      Progress: {progress:.2f}")
            parts.append(f"      Preview: {preview}")
            logger.info("\n".join(parts))

        if self.verbosity >= TRACE:
            logger.info(f"      === FULL GENERATION ({tokens} tokens) ===")
            logger.info(generation)
            logger.info(f"      === END GENERATION ===")
            if feedback:
                logger.info(f"      === FEEDBACK ===")
                logger.info(feedback)
                logger.info(f"      === END FEEDBACK ===")

        # Always write full details to trace file if available
        if self._trace_file:
            self._write_trace({
                "event": "attempt",
                "puzzle": puzzle,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "tokens": tokens,
                "elapsed": elapsed,
                "tok_per_sec": round(tok_per_sec, 1),
                "verified": verified,
                "error": error,
                "progress": progress,
                "generation": generation,
                "feedback": feedback,
            })

    # ── Training update logging (verbosity >= 2) ─────────────────────

    def grpo_advantages(self, advantages: list[tuple[str, float, float]]) -> None:
        """Log GRPO advantage computation.

        Args:
            advantages: list of (puzzle_name, reward, advantage)
        """
        if self.verbosity >= VERBOSE:
            parts = ["  GRPO advantages:"]
            for name, reward, adv in advantages:
                parts.append(f"    {name}: reward={reward:.3f} -> advantage={adv:.2f}")
            logger.info("\n".join(parts))

    def training_update(self, loss: float, num_tokens: int, num_episodes: int) -> None:
        if self.verbosity >= VERBOSE:
            logger.info(
                f"  Policy update: loss={loss:.4f} "
                f"over {num_tokens} tokens from {num_episodes} episodes"
            )

    # ── Utility ──────────────────────────────────────────────────────

    def _write_trace(self, data: dict[str, Any]) -> None:
        if self._trace_file:
            self._trace_file.write(json.dumps(data) + "\n")
            self._trace_file.flush()
