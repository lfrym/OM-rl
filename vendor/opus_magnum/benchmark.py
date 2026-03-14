"""Benchmark harness for evaluating AI models on Opus Magnum puzzles.

Two modes:
  1. Single-shot: model sees the puzzle once and produces a solution.
  2. Iterative: model can submit solutions, see verification feedback,
     and refine to optimize metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .models import Puzzle, Solution
from .puzzle_parser import parse_puzzle
from .solution_writer import write_solution
from .text_format import SOLUTION_FORMAT_SPEC, parse_text_solution, puzzle_to_text
from .verifier import Metrics, Verifier, VerificationError


# ── Model interface ─────────────────────────────────────────────────────────

class ModelInterface(Protocol):
    """Protocol that AI models must implement."""

    def generate_solution(self, prompt: str) -> str:
        """Given a prompt describing the puzzle, return a text solution."""
        ...


# ── Result types ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class AttemptResult:
    """Result of a single solution attempt."""
    success: bool
    error: str | None = None
    metrics: Metrics | None = None
    raw_response: str = ""
    attempt_number: int = 1


@dataclass(slots=True)
class PuzzleResult:
    """Full result for one puzzle."""
    puzzle_name: str
    puzzle_path: str
    solved: bool
    best_metrics: Metrics | None = None
    attempts: list[AttemptResult] = field(default_factory=list)
    total_time_s: float = 0.0


@dataclass(slots=True)
class BenchmarkResult:
    """Full result for a benchmark run."""
    model_name: str
    mode: str
    puzzle_results: list[PuzzleResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def solved_count(self) -> int:
        return sum(1 for r in self.puzzle_results if r.solved)

    @property
    def total_count(self) -> int:
        return len(self.puzzle_results)

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Mode: {self.mode}",
            f"Solved: {self.solved_count}/{self.total_count}",
            f"Total time: {self.total_time_s:.1f}s",
        ]
        if self.solved_count > 0:
            solved = [r for r in self.puzzle_results if r.solved and r.best_metrics]
            avg_cost = sum(r.best_metrics.cost for r in solved) / len(solved)
            avg_cycles = sum(r.best_metrics.cycles for r in solved) / len(solved)
            avg_area = sum(r.best_metrics.area for r in solved) / len(solved)
            lines.extend([
                f"Avg cost: {avg_cost:.0f}",
                f"Avg cycles: {avg_cycles:.0f}",
                f"Avg area: {avg_area:.0f}",
            ])
        return "\n".join(lines)

    def to_json(self) -> str:
        def serialize(obj):
            if isinstance(obj, Metrics):
                return {"cost": obj.cost, "cycles": obj.cycles, "area": obj.area, "instructions": obj.instructions}
            raise TypeError(f"Not serializable: {type(obj)}")
        return json.dumps(asdict(self), default=serialize, indent=2)


# ── Evaluation helpers ──────────────────────────────────────────────────────

def _build_prompt(puzzle: Puzzle, mode: str, feedback: str | None = None) -> str:
    """Build the prompt to send to the model."""
    parts = [
        "You are solving an Opus Magnum puzzle. Opus Magnum is an alchemy puzzle game",
        "on a hexagonal grid. You must place arms, glyphs, tracks, and program arm",
        "instructions to transform input reagents into output products.",
        "",
        "COORDINATE SYSTEM:",
        "The board uses axial hex coordinates (u, v).",
        "Six directions from any hex: E=(+1,0) SE=(0,+1) SW=(-1,+1) W=(-1,0) NW=(0,-1) NE=(+1,-1)",
        "Rotation 0 points in the +u direction (East). Each rotation step is 60° clockwise.",
        "",
        "ARM MECHANICS:",
        "- Arms rotate around their base position. The gripper is at distance 'ext' in the arm's rotation direction.",
        "- Grab (G) picks up the atom/molecule at the gripper position.",
        "- Drop (g) releases what's held.",
        "- Rotate CW/CCW (R/r) swings the gripper 60° around the base.",
        "- Extend/Retract (E/e) changes the gripper distance by 1 (piston only).",
        "- Pivot CW/CCW (P/p) rotates the held molecule around the gripper.",
        "- Forward/Back (A/a) moves the arm along its track.",
        "- Reset (X) returns the arm to the start of its track.",
        "- Repeat (C) loops the instruction tape from the beginning.",
        "",
        "CYCLE EXECUTION:",
        "Each cycle has two half-cycles. In half 1: grab/drop execute. In half 2: movement executes.",
        "After each half-cycle: inputs spawn, glyphs activate, outputs are checked.",
        "Collisions (atoms overlapping, hitting arm bases) halt the machine.",
        "",
        "GLYPH MECHANICS:",
        "- Bonding glyphs create bonds between atoms on their tiles (atoms must not be grabbed).",
        "- Calcification: any cardinal element on the glyph -> salt.",
        "- Projection: quicksilver on tile 0 + metal on tile 1 -> next metal (consumes quicksilver).",
        "- Purification: same metal on tiles 0 and 1 -> next metal on tile 2.",
        "- Animismus: salt on tiles -> produces vitae and mors.",
        "- Unification: air, fire, water, earth on 4 tiles -> quintessence at center.",
        "- Dispersion: quintessence at center -> air, fire, water, earth on 4 tiles.",
        "- Disposal: any atom on the glyph is destroyed.",
        "",
        SOLUTION_FORMAT_SPEC,
        "",
        puzzle_to_text(puzzle),
    ]

    if mode == "iterative" and feedback:
        parts.extend([
            "",
            "PREVIOUS ATTEMPT FEEDBACK:",
            feedback,
            "",
            "Please fix the issues and submit an improved solution.",
        ])
    elif mode == "single_shot":
        parts.append("")
        parts.append("Produce a working solution. Optimize for correctness first, then cost.")
    else:
        parts.append("")
        parts.append("Produce a working solution.")

    return "\n".join(parts)


def _try_verify(puzzle: Puzzle, solution: Solution) -> AttemptResult:
    """Attempt to verify a solution against a puzzle."""
    try:
        solution_bytes = write_solution(solution)
        metrics = Verifier(puzzle.raw_bytes, solution_bytes).evaluate()
        return AttemptResult(success=True, metrics=metrics)
    except VerificationError as e:
        return AttemptResult(success=False, error=str(e))
    except Exception as e:
        return AttemptResult(success=False, error=f"Parse/write error: {e}")


# ── Single-shot mode ────────────────────────────────────────────────────────

def run_single_shot(
    puzzle_path: str | Path,
    model: ModelInterface,
) -> PuzzleResult:
    """Run a single puzzle in single-shot mode."""
    puzzle = parse_puzzle(puzzle_path)
    result = PuzzleResult(puzzle_name=puzzle.name, puzzle_path=str(puzzle_path), solved=False)

    start = time.monotonic()
    prompt = _build_prompt(puzzle, "single_shot")
    response = model.generate_solution(prompt)

    attempt = AttemptResult(success=False, raw_response=response, attempt_number=1)
    try:
        solution = parse_text_solution(response, puzzle.name)
        attempt = _try_verify(puzzle, solution)
        attempt.raw_response = response
        attempt.attempt_number = 1
    except Exception as e:
        attempt.error = f"Failed to parse model response: {e}"

    result.attempts.append(attempt)
    if attempt.success:
        result.solved = True
        result.best_metrics = attempt.metrics
    result.total_time_s = time.monotonic() - start
    return result


# ── Iterative mode ──────────────────────────────────────────────────────────

def run_iterative(
    puzzle_path: str | Path,
    model: ModelInterface,
    max_attempts: int = 5,
) -> PuzzleResult:
    """Run a puzzle in iterative mode with feedback."""
    puzzle = parse_puzzle(puzzle_path)
    result = PuzzleResult(puzzle_name=puzzle.name, puzzle_path=str(puzzle_path), solved=False)

    start = time.monotonic()
    feedback = None

    for attempt_num in range(1, max_attempts + 1):
        prompt = _build_prompt(puzzle, "iterative", feedback)
        response = model.generate_solution(prompt)

        attempt = AttemptResult(
            success=False, raw_response=response, attempt_number=attempt_num,
        )
        try:
            solution = parse_text_solution(response, puzzle.name)
            attempt = _try_verify(puzzle, solution)
            attempt.raw_response = response
            attempt.attempt_number = attempt_num
        except Exception as e:
            attempt.error = f"Failed to parse model response: {e}"

        result.attempts.append(attempt)

        if attempt.success:
            result.solved = True
            m = attempt.metrics
            if result.best_metrics is None or (m and m.cost < result.best_metrics.cost):
                result.best_metrics = m
            # In iterative mode, give the model its score and ask it to optimize.
            feedback = (
                f"Solution VERIFIED! Metrics: cost={m.cost}, cycles={m.cycles}, "
                f"area={m.area}, instructions={m.instructions}.\n"
                f"Try to improve (lower cost, fewer cycles, or smaller area)."
            )
        else:
            feedback = f"Solution FAILED: {attempt.error}\nPlease fix and resubmit."

    result.total_time_s = time.monotonic() - start
    return result


# ── Batch runner ────────────────────────────────────────────────────────────

def run_benchmark(
    puzzle_dir: str | Path,
    model: ModelInterface,
    model_name: str = "unknown",
    mode: str = "single_shot",
    max_attempts: int = 5,
    puzzle_filter: Callable[[Path], bool] | None = None,
) -> BenchmarkResult:
    """Run a full benchmark across all puzzles in a directory."""
    puzzle_dir = Path(puzzle_dir)
    puzzles = sorted(puzzle_dir.rglob("*.puzzle"))
    if puzzle_filter:
        puzzles = [p for p in puzzles if puzzle_filter(p)]

    result = BenchmarkResult(model_name=model_name, mode=mode)
    start = time.monotonic()

    for ppath in puzzles:
        # Skip production puzzles.
        try:
            puz = parse_puzzle(ppath)
            if puz.is_production:
                continue
        except Exception:
            continue

        if mode == "single_shot":
            pr = run_single_shot(ppath, model)
        else:
            pr = run_iterative(ppath, model, max_attempts=max_attempts)
        result.puzzle_results.append(pr)

    result.total_time_s = time.monotonic() - start
    return result
