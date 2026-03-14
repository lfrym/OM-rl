"""Heuristic complexity evaluator for Opus Magnum puzzles.

Provides two main capabilities:
1. Puzzle difficulty scoring — for curriculum ordering
2. Partial progress scoring — for intermediate rewards on failed solutions

The difficulty score estimates how hard a puzzle is based on the transformations
needed, structural complexity of outputs, and number of distinct operations.

The progress score compares a (failed) solution's simulation trace against the
target outputs to estimate how much of the puzzle has been "completed".
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass

from vendor.opus_magnum.models import Atom, BondType, GlyphType, Molecule, Puzzle
from vendor.opus_magnum.verifier import trace, VerificationError

from om_rl.puzzle_gen.chemistry import (
    CARDINALS,
    GLYPH_TO_TOOL,
    METAL_CHAIN,
    find_synthesis_path,
    transformation_depth,
)

logger = logging.getLogger(__name__)


# ── Puzzle Difficulty Scoring ────────────────────────────────────────────────


@dataclass(frozen=True)
class DifficultyScore:
    """Difficulty breakdown for a puzzle."""

    atom_count: int  # Total atoms across all output molecules
    bond_count: int  # Total bonds across all output molecules
    transformation_depth: int  # Min glyph activations needed (sum over all output atoms)
    glyph_variety: int  # Number of distinct glyph types needed
    num_outputs: int  # Number of distinct output molecules
    has_triplex: bool  # Whether any output has triplex bonds
    output_scale: int  # How many copies to produce

    @property
    def score(self) -> float:
        """Combined difficulty score (0-1 normalized, higher = harder).

        Calibrated so campaign chapter 1 puzzles score ~0.1-0.3,
        chapter 3+ puzzles score ~0.5-0.8.
        """
        # Weighted components (each roughly 0-1)
        atom_factor = min(self.atom_count / 12.0, 1.0)
        bond_factor = min(self.bond_count / 8.0, 1.0)
        transform_factor = min(self.transformation_depth / 10.0, 1.0)
        glyph_factor = min(self.glyph_variety / 5.0, 1.0)
        output_factor = min((self.num_outputs - 1) / 3.0, 1.0)
        triplex_factor = 1.0 if self.has_triplex else 0.0

        return (
            0.25 * atom_factor
            + 0.15 * bond_factor
            + 0.30 * transform_factor
            + 0.15 * glyph_factor
            + 0.10 * output_factor
            + 0.05 * triplex_factor
        )


def evaluate_difficulty(puzzle: Puzzle) -> DifficultyScore:
    """Evaluate the difficulty of a puzzle."""
    # Collect all output atoms
    all_output_atoms: list[Atom] = []
    total_bonds = 0
    has_triplex = False

    for mol in puzzle.outputs:
        for _, atom_type in mol.atoms:
            all_output_atoms.append(atom_type)
        total_bonds += len(mol.bonds)
        if any(b.bond_type == BondType.TRIPLEX for b in mol.bonds):
            has_triplex = True

    # Determine available input atoms
    input_atoms: set[Atom] = set()
    for mol in puzzle.inputs:
        for _, atom_type in mol.atoms:
            input_atoms.add(atom_type)

    # Compute transformation depth
    depth = transformation_depth(all_output_atoms, frozenset(input_atoms))
    if depth < 0:
        depth = len(all_output_atoms)  # Fallback if unreachable

    # Count distinct glyph types needed
    glyphs_needed: set[GlyphType] = set()
    for atom in all_output_atoms:
        path = find_synthesis_path(atom, frozenset(input_atoms))
        if path:
            for t in path:
                glyphs_needed.add(t.glyph)
    if total_bonds > 0:
        glyphs_needed.add(GlyphType.BONDER)
    if has_triplex:
        glyphs_needed.add(GlyphType.TRIPLEX_BONDER)

    return DifficultyScore(
        atom_count=len(all_output_atoms),
        bond_count=total_bonds,
        transformation_depth=depth,
        glyph_variety=len(glyphs_needed),
        num_outputs=len(puzzle.outputs),
        has_triplex=has_triplex,
        output_scale=puzzle.output_scale,
    )


# ── Partial Progress Scoring ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ProgressScore:
    """How much progress a (failed) solution made toward the target."""

    # What fraction of target atom types are present on the board at the end
    atom_type_coverage: float
    # What fraction of target atom count is present
    atom_count_coverage: float
    # How many cycles the solution ran before failing (normalized)
    cycle_progress: float
    # Whether the solution was at least parseable and loadable
    structurally_valid: bool

    @property
    def score(self) -> float:
        """Combined progress score (0-1, higher = more progress).

        Returns 0 if the solution wasn't even structurally valid.
        """
        if not self.structurally_valid:
            return 0.0

        return (
            0.30 * self.atom_type_coverage
            + 0.40 * self.atom_count_coverage
            + 0.20 * self.cycle_progress
            + 0.10  # Base credit for being parseable
        )


def evaluate_progress(
    puzzle: Puzzle,
    solution_bytes: bytes,
    max_trace_cycles: int = 50,
) -> ProgressScore:
    """Evaluate how much progress a solution made toward the puzzle's targets.

    Runs a short simulation trace and checks what atoms exist on the board,
    comparing against what the target outputs need.

    Args:
        puzzle: The puzzle being solved.
        solution_bytes: Binary solution bytes.
        max_trace_cycles: How many cycles to simulate for the trace.
    """
    # Collect target atom type counts
    target_atoms: Counter[int] = Counter()
    for mol in puzzle.outputs:
        for _, atom_type in mol.atoms:
            target_atoms[int(atom_type)] += 1

    target_types = set(target_atoms.keys())
    target_total = sum(target_atoms.values())

    # Try to trace the solution
    try:
        states = trace(
            puzzle.raw_bytes,
            solution_bytes,
            max_cycles=max_trace_cycles,
        )
    except VerificationError:
        # Even trace failed — solution has fundamental issues but was loadable
        return ProgressScore(
            atom_type_coverage=0.0,
            atom_count_coverage=0.0,
            cycle_progress=0.0,
            structurally_valid=True,
        )
    except Exception:
        return ProgressScore(
            atom_type_coverage=0.0,
            atom_count_coverage=0.0,
            cycle_progress=0.0,
            structurally_valid=False,
        )

    if not states:
        return ProgressScore(
            atom_type_coverage=0.0,
            atom_count_coverage=0.0,
            cycle_progress=0.0,
            structurally_valid=True,
        )

    # Analyze the last traced cycle for atom presence
    last_state = states[-1]
    board_atoms: Counter[int] = Counter()
    for atom in last_state.get("atoms", []):
        atom_type = atom.get("type")
        if atom_type is not None:
            board_atoms[atom_type] += 1

    # Atom type coverage: what fraction of needed types are on the board?
    if target_types:
        types_present = target_types & set(board_atoms.keys())
        atom_type_coverage = len(types_present) / len(target_types)
    else:
        atom_type_coverage = 0.0

    # Atom count coverage: for each target type, min(board_count, target_count) / target_count
    if target_total > 0:
        matched = sum(
            min(board_atoms.get(t, 0), count) for t, count in target_atoms.items()
        )
        atom_count_coverage = matched / target_total
    else:
        atom_count_coverage = 0.0

    # Cycle progress: how far did it get (normalized by trace length)
    cycle_progress = len(states) / max_trace_cycles

    return ProgressScore(
        atom_type_coverage=atom_type_coverage,
        atom_count_coverage=atom_count_coverage,
        cycle_progress=cycle_progress,
        structurally_valid=True,
    )
