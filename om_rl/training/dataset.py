"""Puzzle dataset management with curriculum learning.

Manages a pool of puzzles (both campaign and generated) and provides
curriculum-based sampling for training.
"""

from __future__ import annotations

import glob
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from vendor.opus_magnum.models import Puzzle
from vendor.opus_magnum.puzzle_parser import parse_puzzle

from om_rl.puzzle_gen.generator import generate_puzzle
from om_rl.puzzle_gen.validator import validate_puzzle
from .config import CurriculumConfig

logger = logging.getLogger(__name__)


@dataclass
class PuzzlePool:
    """A pool of puzzles for training, with curriculum support."""

    campaign_puzzles: list[Puzzle] = field(default_factory=list)
    generated_puzzles: dict[int, list[Puzzle]] = field(default_factory=dict)  # level -> puzzles
    current_level: int = 1
    _rng: random.Random = field(default_factory=random.Random)

    def load_campaign_puzzles(self, puzzle_dir: str) -> int:
        """Load campaign puzzles from a directory."""
        pattern = str(Path(puzzle_dir) / "*.puzzle")
        files = sorted(glob.glob(pattern))
        self.campaign_puzzles = []
        for f in files:
            try:
                p = parse_puzzle(f)
                if not p.is_production:  # Skip production puzzles
                    self.campaign_puzzles.append(p)
            except Exception as e:
                logger.warning(f"Failed to parse {f}: {e}")
        logger.info(f"Loaded {len(self.campaign_puzzles)} campaign puzzles")
        return len(self.campaign_puzzles)

    def generate_puzzles(self, level: int, count: int, base_seed: int = 0) -> int:
        """Generate puzzles for a specific complexity level."""
        if level not in self.generated_puzzles:
            self.generated_puzzles[level] = []

        generated = 0
        for i in range(count):
            try:
                p = generate_puzzle(complexity_level=level, seed=base_seed + i)
                result = validate_puzzle(p)
                if result:
                    self.generated_puzzles[level].append(p)
                    generated += 1
                else:
                    logger.debug(f"Generated puzzle failed validation: {result.issues}")
            except Exception as e:
                logger.debug(f"Failed to generate puzzle: {e}")

        logger.info(f"Generated {generated}/{count} valid puzzles at level {level}")
        return generated

    def sample(
        self,
        n: int,
        config: CurriculumConfig | None = None,
    ) -> list[Puzzle]:
        """Sample n puzzles according to curriculum settings."""
        if config is None:
            config = CurriculumConfig()

        puzzles: list[Puzzle] = []
        for _ in range(n):
            use_generated = self._rng.random() < config.generated_ratio

            if use_generated and self.current_level in self.generated_puzzles:
                pool = self.generated_puzzles[self.current_level]
                if pool:
                    puzzles.append(self._rng.choice(pool))
                    continue

            # Fall back to campaign puzzles
            if self.campaign_puzzles:
                puzzles.append(self._rng.choice(self.campaign_puzzles))
            elif self.generated_puzzles:
                # Use any generated puzzles available
                all_gen = [p for ps in self.generated_puzzles.values() for p in ps]
                if all_gen:
                    puzzles.append(self._rng.choice(all_gen))

        return puzzles

    def maybe_advance_level(
        self,
        solve_rate: float,
        config: CurriculumConfig,
    ) -> bool:
        """Check if we should advance to the next curriculum level.

        Returns True if level was advanced.
        """
        if solve_rate >= config.advance_threshold:
            max_level = max(config.levels)
            if self.current_level < max_level:
                self.current_level += 1
                logger.info(
                    f"Curriculum: advancing to level {self.current_level} "
                    f"(solve_rate={solve_rate:.2%} >= {config.advance_threshold:.2%})"
                )
                return True
        return False

    @property
    def total_puzzles(self) -> int:
        gen = sum(len(ps) for ps in self.generated_puzzles.values())
        return len(self.campaign_puzzles) + gen
