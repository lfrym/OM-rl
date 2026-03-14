"""Tinker-compatible environment for Opus Magnum RL.

Adapts our puzzle generator, omsim verifier, and structure scorer
into Tinker's ProblemEnv interface. This lets us train via Tinker's
infrastructure while reusing all our existing puzzle/verification logic.

Tinker's ProblemEnv expects:
  - get_question() -> str: the puzzle prompt
  - check_answer(response: str) -> bool: did the model solve it?
  - check_format(response: str) -> bool: is the response well-formed?
  - get_reference_answer() -> str: a known-good answer for logging

We extend this with our structure scorer for richer reward signal.
"""

from __future__ import annotations

import logging
from typing import Any

from vendor.opus_magnum.models import Puzzle
from vendor.opus_magnum.text_format import parse_text_solution
from vendor.opus_magnum.solution_writer import write_solution
from vendor.opus_magnum.verifier import Verifier, VerificationError

from om_rl.env.observation import (
    GAME_REFERENCE,
    SOLUTION_FORMAT,
    FEW_SHOT_EXAMPLES,
)
from om_rl.env.reward import RewardConfig
from om_rl.complexity.structure_scorer import score_solution_structure
from om_rl.puzzle_gen.generator import generate_puzzle
from om_rl.puzzle_gen.validator import validate_puzzle

logger = logging.getLogger(__name__)

# The reference solution we use for logging (verified correct for seed=1)
REFERENCE_SOLUTION = """\
INPUT pos=(-2,0) rot=0 idx=0
OUTPUT pos=(2,0) rot=0 idx=0
GLYPH glyph-calcification pos=(0,0) rot=0
ARM arm1 pos=(-1,0) rot=3 ext=1 id=0
  TAPE: 1:G 2:R 3:R 4:R 5:g 6:R 7:R 8:R 9:C
ARM arm1 pos=(1,0) rot=3 ext=1 id=1
  TAPE: 7:G 8:R 9:R 10:R 11:g 12:R 13:R 14:R 15:C"""


def _format_puzzle_prompt(puzzle: Puzzle) -> str:
    """Format a puzzle as a prompt string (without system-level framing).

    Tinker's ProblemEnv wraps this in its own conversation format.
    """
    from vendor.opus_magnum.text_format import puzzle_to_text

    puzzle_text = puzzle_to_text(puzzle)

    num_inputs = len(puzzle.inputs)
    num_outputs = len(puzzle.outputs)
    input_indices = ", ".join(str(i) for i in range(num_inputs))
    output_indices = ", ".join(str(i) for i in range(num_outputs))

    io_text = (
        f"I/O PLACEMENT:\n"
        f"Place inputs and outputs anywhere on the board.\n"
        f"  Required inputs: {num_inputs} (molecule indices: {input_indices})\n"
        f"  Required outputs: {num_outputs} (molecule indices: {output_indices})\n"
        f"Arms grab atoms FROM inputs and deliver assembled molecules TO outputs."
    )

    return (
        f"Solve this Opus Magnum puzzle. Output a complete solution.\n\n"
        f"{GAME_REFERENCE}\n"
        f"{SOLUTION_FORMAT}\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        f"{puzzle_text}\n"
        f"{io_text}\n"
        f"Output ONLY the solution — no explanation needed."
    )


def _verify_with_omsim(
    solution_text: str,
    puzzle: Puzzle,
    cycle_limit: int = 100_000,
) -> tuple[bool, str | None, int, dict[str, int] | None]:
    """Try to parse and verify a solution.

    Returns: (verified, error_message, error_cycle, metrics_dict)
    """
    try:
        solution = parse_text_solution(solution_text, puzzle.name)
    except Exception as e:
        return False, f"Parse error: {e}", -1, None

    try:
        sol_bytes = write_solution(solution)
        with Verifier(puzzle.raw_bytes, sol_bytes, cycle_limit) as v:
            metrics = v.evaluate()
        return True, None, -1, {
            "cost": metrics.cost,
            "cycles": metrics.cycles,
            "area": metrics.area,
            "instructions": metrics.instructions,
        }
    except VerificationError as e:
        return False, str(e), e.cycle, None
    except Exception as e:
        return False, f"Internal error: {e}", -1, None


class OpusMagnumPuzzleEnv:
    """Tinker ProblemEnv-compatible environment for Opus Magnum.

    This class implements the interface expected by tinker_cookbook's
    ProblemEnv / ProblemGroupBuilder pattern. It can be used directly
    with Tinker's training infrastructure.

    If tinker_cookbook is installed, use OpusMagnumTinkerEnv which
    inherits from ProblemEnv. If not, this standalone class has the
    same methods for testing.
    """

    def __init__(
        self,
        puzzle: Puzzle,
        cycle_limit: int = 100_000,
        use_structure_scoring: bool = True,
        format_coef: float = 0.1,
    ):
        self.puzzle = puzzle
        self.cycle_limit = cycle_limit
        self.use_structure_scoring = use_structure_scoring
        self.format_coef = format_coef
        self._question = _format_puzzle_prompt(puzzle)

    def get_question(self) -> str:
        return self._question

    def check_answer(self, response: str) -> bool | float:
        """Check if the response solves the puzzle.

        Returns True/False for binary reward, or a float 0-1 for
        structure-scored partial credit.
        """
        verified, error, error_cycle, metrics = _verify_with_omsim(
            response, self.puzzle, self.cycle_limit
        )

        if verified:
            return True

        if self.use_structure_scoring:
            struct = score_solution_structure(
                response, self.puzzle,
                omsim_error=error,
                omsim_error_cycle=error_cycle,
            )
            return struct.score

        return False

    def check_format(self, response: str) -> bool:
        """Check if the response has valid solution format."""
        lines = response.strip().split("\n")
        has_input = any(l.strip().startswith("INPUT ") for l in lines)
        has_output = any(l.strip().startswith("OUTPUT ") for l in lines)
        has_arm = any(l.strip().startswith("ARM ") for l in lines)
        return has_input and has_output and has_arm

    def get_reference_answer(self) -> str:
        return REFERENCE_SOLUTION


class OpusMagnumGroupBuilder:
    """Builds groups of puzzle environments.

    Compatible with Tinker's EnvGroupBuilder pattern. Each group
    contains multiple environments for the same puzzle (for GRPO-style
    group comparison).
    """

    def __init__(
        self,
        puzzle: Puzzle,
        group_size: int = 4,
        cycle_limit: int = 100_000,
        use_structure_scoring: bool = True,
    ):
        self.puzzle = puzzle
        self.group_size = group_size
        self.cycle_limit = cycle_limit
        self.use_structure_scoring = use_structure_scoring

    def make_envs(self) -> list[OpusMagnumPuzzleEnv]:
        """Create a group of environments for GRPO."""
        return [
            OpusMagnumPuzzleEnv(
                puzzle=self.puzzle,
                cycle_limit=self.cycle_limit,
                use_structure_scoring=self.use_structure_scoring,
            )
            for _ in range(self.group_size)
        ]


class OpusMagnumDatasetBuilder:
    """Builds the RL dataset from generated and campaign puzzles.

    Compatible with Tinker's RLDatasetBuilder pattern.
    """

    def __init__(
        self,
        complexity_level: int = 1,
        batch_size: int = 128,
        group_size: int = 16,
        num_puzzles: int = 1000,
        campaign_puzzle_dir: str = "puzzles/campaign",
        generated_ratio: float = 0.7,
        seed: int = 42,
        cycle_limit: int = 100_000,
        use_structure_scoring: bool = True,
    ):
        self.complexity_level = complexity_level
        self.batch_size = batch_size
        self.group_size = group_size
        self.num_puzzles = num_puzzles
        self.campaign_puzzle_dir = campaign_puzzle_dir
        self.generated_ratio = generated_ratio
        self.seed = seed
        self.cycle_limit = cycle_limit
        self.use_structure_scoring = use_structure_scoring
        self._puzzles: list[Puzzle] | None = None

    def _load_puzzles(self) -> list[Puzzle]:
        """Load/generate the puzzle pool."""
        if self._puzzles is not None:
            return self._puzzles

        import glob
        import random
        from vendor.opus_magnum.puzzle_parser import parse_puzzle

        rng = random.Random(self.seed)
        puzzles: list[Puzzle] = []

        # Generate puzzles
        num_generated = int(self.num_puzzles * self.generated_ratio)
        for i in range(num_generated):
            try:
                p = generate_puzzle(
                    complexity_level=self.complexity_level,
                    seed=self.seed + i,
                )
                if validate_puzzle(p):
                    puzzles.append(p)
            except Exception:
                pass

        # Load campaign puzzles
        pattern = f"{self.campaign_puzzle_dir}/*.puzzle"
        for f in sorted(glob.glob(pattern)):
            try:
                p = parse_puzzle(f)
                if not p.is_production:
                    puzzles.append(p)
            except Exception:
                pass

        rng.shuffle(puzzles)
        self._puzzles = puzzles
        logger.info(f"Loaded {len(puzzles)} puzzles "
                    f"({num_generated} generated, {len(puzzles) - num_generated} campaign)")
        return puzzles

    def get_batch(self, index: int) -> list[OpusMagnumGroupBuilder]:
        """Get a batch of EnvGroupBuilders for training."""
        puzzles = self._load_puzzles()
        batch: list[OpusMagnumGroupBuilder] = []

        for i in range(self.batch_size):
            puzzle_idx = (index * self.batch_size + i) % len(puzzles)
            batch.append(OpusMagnumGroupBuilder(
                puzzle=puzzles[puzzle_idx],
                group_size=self.group_size,
                cycle_limit=self.cycle_limit,
                use_structure_scoring=self.use_structure_scoring,
            ))

        return batch
