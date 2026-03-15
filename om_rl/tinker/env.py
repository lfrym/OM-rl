"""Tinker-compatible environment for Opus Magnum RL.

Adapts our puzzle generator, omsim verifier, and structure scorer
into Tinker's ProblemEnv interface.

Tinker's ProblemEnv.step() computes reward as:
  format_coef * (check_format - 1) + check_answer

So if check_answer returns a float (structure score 0-1), the reward
is smooth. If check_format returns 0, there's a format penalty.
"""

from __future__ import annotations

import glob
import logging
import math
import random
from typing import Literal, Sequence

from vendor.opus_magnum.models import Puzzle
from vendor.opus_magnum.text_format import parse_text_solution
from vendor.opus_magnum.solution_writer import write_solution
from vendor.opus_magnum.verifier import Verifier, VerificationError

from om_rl.env.observation import (
    GAME_REFERENCE,
    SOLUTION_FORMAT,
    FEW_SHOT_EXAMPLES,
)
from om_rl.complexity.structure_scorer import score_solution_structure
from om_rl.puzzle_gen.generator import generate_puzzle
from om_rl.puzzle_gen.validator import validate_puzzle

logger = logging.getLogger(__name__)

# Verified reference solution for logging
REFERENCE_SOLUTION = """\
INPUT pos=(-2,0) rot=0 idx=0
OUTPUT pos=(2,0) rot=0 idx=0
GLYPH glyph-calcification pos=(0,0) rot=0
ARM arm1 pos=(-1,0) rot=3 ext=1 id=0
  TAPE: 1:G 2:R 3:R 4:R 5:g 6:R 7:R 8:R 9:C
ARM arm1 pos=(1,0) rot=3 ext=1 id=1
  TAPE: 7:G 8:R 9:R 10:R 11:g 12:R 13:R 14:R 15:C"""


def _format_puzzle_prompt(puzzle: Puzzle) -> str:
    """Format a puzzle as a prompt string."""
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
        f"{GAME_REFERENCE}\n{SOLUTION_FORMAT}\n{FEW_SHOT_EXAMPLES}\n"
        f"{puzzle_text}\n{io_text}\n"
        f"Output ONLY the solution — no explanation needed."
    )


def _verify_with_omsim(
    solution_text: str, puzzle: Puzzle, cycle_limit: int = 100_000,
) -> tuple[bool, str | None, int]:
    """Parse and verify a solution. Returns (verified, error, error_cycle)."""
    try:
        solution = parse_text_solution(solution_text, puzzle.name)
    except Exception as e:
        return False, f"Parse error: {e}", -1

    try:
        sol_bytes = write_solution(solution)
        with Verifier(puzzle.raw_bytes, sol_bytes, cycle_limit) as v:
            v.evaluate()
        return True, None, -1
    except VerificationError as e:
        return False, str(e), e.cycle
    except Exception as e:
        return False, f"Internal error: {e}", -1


def _build_puzzle_pool(
    complexity_level: int,
    num_puzzles: int,
    campaign_puzzle_dir: str,
    generated_ratio: float,
    seed: int,
) -> list[Puzzle]:
    """Load/generate a puzzle pool."""
    from vendor.opus_magnum.puzzle_parser import parse_puzzle as parse_puzzle_file

    rng = random.Random(seed)
    puzzles: list[Puzzle] = []

    num_generated = int(num_puzzles * generated_ratio)
    for i in range(num_generated):
        try:
            p = generate_puzzle(complexity_level=complexity_level, seed=seed + i)
            if validate_puzzle(p):
                puzzles.append(p)
        except Exception:
            pass

    for f in sorted(glob.glob(f"{campaign_puzzle_dir}/*.puzzle")):
        try:
            p = parse_puzzle_file(f)
            if not p.is_production:
                puzzles.append(p)
        except Exception:
            pass

    rng.shuffle(puzzles)
    logger.info(f"Puzzle pool: {len(puzzles)} puzzles (level={complexity_level})")
    return puzzles


def make_tinker_dataset_builder(
    complexity_level: int = 1,
    max_level: int | None = None,
    curriculum_step_interval: int = 10,
    batch_size: int = 128,
    group_size: int = 16,
    num_puzzles: int = 1000,
    campaign_puzzle_dir: str = "puzzles/campaign",
    generated_ratio: float = 0.7,
    seed: int = 42,
    cycle_limit: int = 100_000,
    use_structure_scoring: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
    renderer_name: str = "qwen3",
):
    """Build a Tinker RLDatasetBuilder for Opus Magnum.

    Args:
        complexity_level: Starting puzzle level (1-5).
        max_level: Maximum level for curriculum. If None, stays at complexity_level.
            If set, puzzles ramp from complexity_level to max_level over training.
        curriculum_step_interval: Steps between level increases.
            E.g., interval=10 means level increases every 10 steps.
            At each step, the mix is weighted toward the current curriculum level
            with some easier puzzles mixed in for reinforcement.
        batch_size: Puzzles per step.
        group_size: Completions per puzzle (K for GRPO).

    Returns an instance compatible with train.Config.dataset_builder.
    """
    from tinker_cookbook import renderers
    from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
    from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder, EnvGroupBuilder
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    import chz

    if max_level is None:
        max_level = complexity_level

    # Pre-generate puzzle pools for each level
    puzzle_pools: dict[int, list[Puzzle]] = {}
    for lvl in range(complexity_level, max_level + 1):
        puzzle_pools[lvl] = _build_puzzle_pool(
            lvl, num_puzzles, campaign_puzzle_dir,
            generated_ratio, seed + lvl * 10000,
        )

    # Define the ProblemEnv subclass
    class OpusMagnumEnv(ProblemEnv):
        def __init__(
            self,
            puzzle: Puzzle,
            renderer: renderers.Renderer,
            _cycle_limit: int = cycle_limit,
            _use_structure_scoring: bool = use_structure_scoring,
        ):
            super().__init__(renderer)
            self.puzzle = puzzle
            self._cycle_limit = _cycle_limit
            self._use_structure_scoring = _use_structure_scoring

        def get_question(self) -> str:
            return _format_puzzle_prompt(self.puzzle)

        def check_answer(self, response: str) -> float:
            verified, error, error_cycle = _verify_with_omsim(
                response, self.puzzle, self._cycle_limit
            )
            if verified:
                return 1.0
            if self._use_structure_scoring:
                struct = score_solution_structure(
                    response, self.puzzle,
                    omsim_error=error,
                    omsim_error_cycle=error_cycle,
                )
                return struct.score
            return 0.0

        def check_format(self, response: str) -> bool:
            lines = response.strip().split("\n")
            has_input = any(l.strip().startswith("INPUT ") for l in lines)
            has_output = any(l.strip().startswith("OUTPUT ") for l in lines)
            has_arm = any(l.strip().startswith("ARM ") for l in lines)
            return has_input and has_output and has_arm

        def get_reference_answer(self) -> str:
            return REFERENCE_SOLUTION

    # Define the Dataset
    class OpusMagnumDataset(RLDataset):
        def __init__(self, renderer: renderers.Renderer):
            self.renderer = renderer

        def _get_curriculum_level(self, step: int) -> int:
            """Determine current max curriculum level based on step."""
            levels_unlocked = step // curriculum_step_interval
            return min(complexity_level + levels_unlocked, max_level)

        def _pick_puzzle(self, step: int, idx: int) -> Puzzle:
            """Pick a puzzle with curriculum-weighted level selection.

            At each step, we sample from all unlocked levels with a
            distribution weighted toward the current (hardest unlocked)
            level: 50% current level, 50% uniform across all unlocked.
            """
            current_max = self._get_curriculum_level(step)
            rng = random.Random(seed + step * 1000 + idx)

            if rng.random() < 0.5 or current_max == complexity_level:
                # Sample from current (hardest unlocked) level
                lvl = current_max
            else:
                # Sample uniformly from all unlocked levels
                lvl = rng.randint(complexity_level, current_max)

            pool = puzzle_pools[lvl]
            return pool[(step * batch_size + idx) % len(pool)]

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            batch = []
            current_lvl = self._get_curriculum_level(index)
            if index % 5 == 0:  # Log curriculum level every 5 steps
                logger.info(f"  Curriculum: step {index}, max_level={current_lvl}")

            for i in range(batch_size):
                puzzle = self._pick_puzzle(index, i)

                def make_env(p=puzzle, r=self.renderer):
                    return OpusMagnumEnv(puzzle=p, renderer=r)

                batch.append(ProblemGroupBuilder(
                    env_thunk=make_env,
                    num_envs=group_size,
                ))
            return batch

        def __len__(self) -> int:
            total = sum(len(p) for p in puzzle_pools.values())
            return math.ceil(total / batch_size)

    # Define the DatasetBuilder
    @chz.chz
    class OpusMagnumDatasetBuilder(RLDatasetBuilder):
        model_name_for_tokenizer: str = model_name
        renderer_name_: str = renderer_name

        async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
            tokenizer = get_tokenizer(self.model_name_for_tokenizer)
            renderer = renderers.get_renderer(self.renderer_name_, tokenizer=tokenizer)
            return OpusMagnumDataset(renderer), None

    return OpusMagnumDatasetBuilder()
