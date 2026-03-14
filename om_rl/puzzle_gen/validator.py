"""Puzzle solvability validation.

Checks whether a puzzle's outputs are chemically reachable from its inputs.
This is a necessary condition for solvability (but not sufficient — it doesn't
account for spatial constraints like board layout or arm reach).

Since our generator works backward from outputs, generated puzzles should always
pass this check. This module serves as a sanity check and for validating
externally-sourced puzzles.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from vendor.opus_magnum.models import Atom, GlyphType, Puzzle, ToolFlags

from .chemistry import (
    CARDINALS,
    GLYPH_TO_TOOL,
    METAL_CHAIN,
    find_synthesis_path,
)


@dataclass
class ValidationResult:
    """Result of puzzle validation."""

    is_valid: bool
    issues: list[str]

    def __bool__(self) -> bool:
        return self.is_valid


def validate_puzzle(puzzle: Puzzle) -> ValidationResult:
    """Check whether a puzzle is likely solvable.

    Checks:
    1. Puzzle has at least one input and one output
    2. All output atom types are chemically reachable from input atom types
       using the puzzle's allowed glyphs
    3. Required bonding glyphs are available for bonded outputs
    """
    issues: list[str] = []

    # Basic structure checks
    if not puzzle.inputs:
        issues.append("Puzzle has no inputs")
    if not puzzle.outputs:
        issues.append("Puzzle has no outputs")
    if issues:
        return ValidationResult(False, issues)

    # Collect available input atom types
    available_atoms: set[Atom] = set()
    for mol in puzzle.inputs:
        for _, atom_type in mol.atoms:
            available_atoms.add(atom_type)

    # Check which glyphs are available from tool flags
    available_glyphs: set[GlyphType] = set()
    for glyph, flag in GLYPH_TO_TOOL.items():
        if puzzle.allowed_tools & flag:
            available_glyphs.add(glyph)

    # Check each output atom is reachable
    for i, mol in enumerate(puzzle.outputs):
        for pos, atom_type in mol.atoms:
            if atom_type in available_atoms:
                continue

            path = find_synthesis_path(atom_type, frozenset(available_atoms))
            if path is None:
                issues.append(
                    f"Output {i}: atom {atom_type.name} at ({pos.u},{pos.v}) "
                    f"is not reachable from available inputs"
                )
                continue

            # Check that required glyphs are available
            for t in path:
                if t.glyph not in available_glyphs:
                    issues.append(
                        f"Output {i}: atom {atom_type.name} requires "
                        f"{t.glyph.value} but it's not in allowed tools"
                    )

        # Check bonding glyphs
        for bond in mol.bonds:
            from vendor.opus_magnum.models import BondType
            if bond.bond_type == BondType.NORMAL:
                if not (puzzle.allowed_tools & ToolFlags.BONDER):
                    issues.append(
                        f"Output {i}: has normal bonds but bonder is not available"
                    )
            if bond.bond_type == BondType.TRIPLEX:
                if not (puzzle.allowed_tools & ToolFlags.TRIPLEX_BONDER):
                    issues.append(
                        f"Output {i}: has triplex bonds but triplex bonder is not available"
                    )

    return ValidationResult(len(issues) == 0, issues)
