"""Tinker-compatible environment for Opus Magnum RL.

Adapts our puzzle generator, omsim verifier, and structure scorer
into Tinker's ProblemEnv interface.

Multi-turn: the model gets up to max_attempts tries per puzzle.
Each failed attempt returns episode_done=False with the omsim error
as feedback. Reward is only assigned on the final step.

Final reward formula (same as ProblemEnv default):
  format_coef * (check_format - 1) + check_answer
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

from vendor.opus_magnum.models import Puzzle, Solution
from vendor.opus_magnum.text_format import parse_text_solution
from vendor.opus_magnum.solution_writer import write_solution
from vendor.opus_magnum.verifier import Verifier, VerificationError

from om_rl.env.observation import (
    format_initial_observation,
    format_feedback_observation,
)
from om_rl.complexity.structure_scorer import score_solution_structure
from om_rl.puzzle_gen.generator import generate_puzzle
from om_rl.puzzle_gen.validator import validate_puzzle

logger = logging.getLogger(__name__)

# Reference solution for logging (calcification example, vertical layout)
REFERENCE_SOLUTION = """\
INPUT pos=(0,2) rot=0 idx=0
OUTPUT pos=(0,-2) rot=0 idx=0
GLYPH glyph-calcification pos=(0,0) rot=0
ARM arm1 pos=(0,1) rot=1 ext=1 id=0
  TAPE: 1:G 2:R 3:R 4:R 5:g 6:R 7:R 8:R 9:C
ARM arm1 pos=(0,-1) rot=1 ext=1 id=1
  TAPE: 7:G 8:R 9:R 10:R 11:g 12:R 13:R 14:R 15:C"""


def _format_puzzle_prompt(puzzle: Puzzle) -> str:
    """Format a puzzle as a prompt string."""
    return format_initial_observation(puzzle)


def _verify_with_omsim(
    solution_text: str, puzzle: Puzzle, cycle_limit: int = 100_000,
) -> tuple[bool, str | None, int, tuple[int, int] | None, Solution | None]:
    """Parse and verify a solution.

    Returns (verified, error, error_cycle, error_location, solution).
    solution is None if parsing failed.
    """
    try:
        solution = parse_text_solution(solution_text, puzzle.name)
    except Exception as e:
        return False, f"Parse error: {e}", -1, None, None

    try:
        sol_bytes = write_solution(solution)
        with Verifier(puzzle.raw_bytes, sol_bytes, cycle_limit) as v:
            v.evaluate()
        return True, None, -1, None, solution
    except VerificationError as e:
        error_str = str(e)
        error_cycle = e.cycle
        error_location = e.location if (e.location and e.location != (0, 0)) else None
        # "did not complete" errors report cycle=0 and location=(0,0) as defaults — meaningless
        is_timeout = "cycle limit" in error_str.lower() or "did not complete" in error_str.lower()
        if is_timeout:
            error_cycle = -1
            error_location = None
            error_str = (
                "The machine ran for the maximum number of cycles without completing. "
                "This usually means atoms are not reaching the output position, or the "
                "output molecule shape/orientation does not match."
            )
        return False, error_str, error_cycle, error_location, solution
    except Exception as e:
        return False, f"Internal error: {e}", -1, None, solution


def _build_puzzle_pool(
    complexity_level: int,
    num_puzzles: int,
    seed: int,
) -> list[Puzzle]:
    """Generate a training puzzle pool (no campaign puzzles — those are the test set)."""
    rng = random.Random(seed)
    puzzles: list[Puzzle] = []

    for i in range(num_puzzles):
        try:
            p = generate_puzzle(complexity_level=complexity_level, seed=seed + i)
            if validate_puzzle(p):
                puzzles.append(p)
        except Exception:
            pass

    rng.shuffle(puzzles)
    logger.info(f"Puzzle pool: {len(puzzles)} generated puzzles (level={complexity_level})")
    return puzzles


def make_tinker_dataset_builder(
    complexity_level: int = 1,
    max_level: int | None = None,
    curriculum_step_interval: int = 10,
    batch_size: int = 128,
    group_size: int = 16,
    max_steps: int = 100,
    num_puzzles: int = 1000,
    seed: int = 42,
    cycle_limit: int = 100_000,
    use_structure_scoring: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
    renderer_name: str = "qwen3",
    max_tokens: int = 8192,
    max_attempts: int = 3,
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
    import tinker
    import chz
    from tinker_cookbook import renderers
    from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
    from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder, EnvGroupBuilder, StepResult
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    if max_level is None:
        max_level = complexity_level

    # Reserve ~25% of the token budget for final response
    thinking_budget = (max_tokens * 3) // 4
    solution_budget = max_tokens - thinking_budget

    _system_prompt = (
        f"You are solving Opus Magnum puzzles. You have up to {max_attempts} attempts per puzzle — "
        f"after each failed attempt you will receive the error from the simulator and can revise your solution. "
        f"You do not need to solve the puzzle on the first try: use early attempts to get the structure right "
        f"(valid INPUT/OUTPUT/ARM/TAPE lines), then refine positions and tape logic in later attempts.\n\n"
        f"Token budget: you have {max_tokens} tokens per attempt. "
        f"Keep your reasoning under {thinking_budget} tokens so you have at least "
        f"{solution_budget} tokens left to write the solution. "
        f"If you are still reasoning and running low on space, stop immediately "
        f"and output the best solution you have — a valid attempt that can be scored "
        f"is always better than a truncated response that scores zero."
    )

    # Pre-generate puzzle pools for each level
    puzzle_pools: dict[int, list[Puzzle]] = {}
    for lvl in range(complexity_level, max_level + 1):
        puzzle_pools[lvl] = _build_puzzle_pool(
            lvl, num_puzzles, seed + lvl * 10000,
        )

    # Define the ProblemEnv subclass
    class OpusMagnumEnv(ProblemEnv):
        def __init__(
            self,
            puzzle: Puzzle,
            renderer: renderers.Renderer,
            _cycle_limit: int = cycle_limit,
            _use_structure_scoring: bool = use_structure_scoring,
            _convo_prefix: list | None = None,
        ):
            super().__init__(renderer, convo_prefix=_convo_prefix or [])
            self.puzzle = puzzle
            self._cycle_limit = _cycle_limit
            self._use_structure_scoring = _use_structure_scoring
            self._attempt = 0
            self._convo: list = []

        def get_question(self) -> str:
            return _format_puzzle_prompt(self.puzzle)

        def get_reference_answer(self) -> str:
            return REFERENCE_SOLUTION

        def _score(self, response: str, verified: bool, error: str | None, error_cycle: int) -> tuple[float, int]:
            """Compute (score, level) given pre-run omsim results."""
            if verified:
                return (1.0, 10)
            if self._use_structure_scoring:
                struct = score_solution_structure(
                    response, self.puzzle,
                    omsim_error=error,
                    omsim_error_cycle=error_cycle,
                )
                # Only give credit for L7+ (omsim accepted the solution).
                # L0-L6 get 0.0 — no reward for formatted-but-wrong.
                if struct.level >= 7:
                    return (struct.score, struct.level)
                else:
                    return (0.0, struct.level)
            return (0.0, 0)

        # check_answer / check_format are required by the abstract base but only
        # used in ProblemEnv.step(), which we fully override below.
        def check_answer(self, response: str) -> float:
            verified, error, error_cycle, _, _ = _verify_with_omsim(response, self.puzzle, self._cycle_limit)
            score, _ = self._score(response, verified, error, error_cycle)
            return score

        def check_format(self, response: str) -> bool:
            verified, error, error_cycle, _, _ = _verify_with_omsim(response, self.puzzle, self._cycle_limit)
            _, level = self._score(response, verified, error, error_cycle)
            return level >= 4

        async def initial_observation(self):
            self._attempt = 0
            self._convo = self.convo_prefix + [
                {"role": "user", "content": self.get_question()},
            ]
            return self.renderer.build_generation_prompt(self._convo), self.stop_condition

        async def step(self, action) -> StepResult:
            message, parse_success = self.renderer.parse_response(action)
            response = message["content"]
            self._attempt += 1

            verified, error, error_cycle, error_location, solution = _verify_with_omsim(
                response, self.puzzle, self._cycle_limit
            )
            is_final = verified or self._attempt >= max_attempts

            if is_final:
                score, level = self._score(response, verified, error, error_cycle)
                correct_format = float(parse_success and level >= 4)
                total_reward = self.format_coef * (correct_format - 1) + score
                return StepResult(
                    reward=total_reward,
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={"format": correct_format, "correct": score},
                )
            else:
                # Append assistant response + error feedback, continue episode
                self._convo.append({"role": "assistant", "content": response})
                feedback = format_feedback_observation(
                    solution, self.puzzle, error or "", error_cycle, error_location,
                    self._attempt, max_attempts,
                )
                self._convo.append({"role": "user", "content": feedback})
                next_ob = self.renderer.build_generation_prompt(self._convo)
                return StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=next_ob,
                    next_stop_condition=self.stop_condition,
                    metrics={},
                )

    # Define the Dataset
    class OpusMagnumDataset(RLDataset):
        def __init__(self, renderer: renderers.Renderer):
            self.renderer = renderer

        def _get_curriculum_window(self, step: int) -> tuple[int, int]:
            """Determine the active level window based on step.

            Returns (min_level, max_level) for this step. The window
            slides upward over training:
              Steps 0-4:  L1 only
              Steps 5-9:  L1-L2
              Steps 10-14: L2 only
              Steps 15-19: L2-L3
              Steps 20-24: L3 only
              ...

            Each interval introduces the next level, then the previous
            level is phased out.
            """
            phase = step // curriculum_step_interval
            # Even phases = single level, odd phases = transition (two levels)
            # Phase 0: L1 only, Phase 1: L1+L2, Phase 2: L2 only, Phase 3: L2+L3, ...
            base = complexity_level + phase // 2
            if phase % 2 == 0:
                # Single level phase
                lvl = min(base, max_level)
                return (lvl, lvl)
            else:
                # Transition phase: current + next
                lo = min(base, max_level)
                hi = min(base + 1, max_level)
                return (lo, hi)

        def _pick_puzzle(self, step: int, idx: int) -> Puzzle:
            """Pick a puzzle from the current curriculum window.

            During single-level phases, all puzzles are from that level.
            During transition phases, 50/50 split between the two levels.
            """
            lo, hi = self._get_curriculum_window(step)
            rng = random.Random(seed + step * 1000 + idx)

            if lo == hi:
                lvl = lo
            else:
                lvl = lo if rng.random() < 0.5 else hi

            pool = puzzle_pools[lvl]
            return pool[(step * batch_size + idx) % len(pool)]

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            batch = []
            lo, hi = self._get_curriculum_window(index)
            if index % curriculum_step_interval == 0:
                if lo == hi:
                    logger.info(f"  Curriculum: step {index}, level L{lo}")
                else:
                    logger.info(f"  Curriculum: step {index}, levels L{lo}-L{hi} (transition)")

            for i in range(batch_size):
                puzzle = self._pick_puzzle(index, i)

                def make_env(p=puzzle, r=self.renderer):
                    return OpusMagnumEnv(
                        puzzle=p,
                        renderer=r,
                        _convo_prefix=[{"role": "system", "content": _system_prompt}],
                    )

                batch.append(ProblemGroupBuilder(
                    env_thunk=make_env,
                    num_envs=group_size,
                ))
            return batch

        def __len__(self) -> int:
            return max_steps

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
