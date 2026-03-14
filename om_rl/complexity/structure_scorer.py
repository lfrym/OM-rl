"""Solution structure scorer — grades submissions before omsim verification.

Provides a smooth reward signal by scoring structural correctness at
multiple levels. Each level is a prerequisite for the next:

  Level 0 (0.0): Unparseable — can't extract any solution lines
  Level 1 (0.1): Has at least one valid solution line (ARM, GLYPH, INPUT, or OUTPUT)
  Level 2 (0.2): Has INPUT and OUTPUT lines
  Level 3 (0.3): Has correct number of INPUT/OUTPUT lines for this puzzle
  Level 4 (0.4): Has at least one ARM with a TAPE
  Level 5 (0.5): Has appropriate GLYPH(s) for the required transformation
  Level 6 (0.6): All glyph/arm type names are valid
  Level 7 (0.7): No static overlaps at cycle 0 (positions don't collide)
  Level 8 (0.8): Passes omsim load (structurally valid enough to simulate)
  Level 9 (0.9): Runs for >0 cycles before failing
  Level 10 (1.0): Verified correct by omsim

This can be toggled via RewardConfig.use_structure_scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vendor.opus_magnum.models import (
    ArmType,
    GlyphType,
    Puzzle,
    ToolFlags,
)

# Map from puzzle ToolFlags to which glyph types are needed
_TOOL_TO_GLYPH: dict[ToolFlags, str] = {
    ToolFlags.CALCIFICATION: "glyph-calcification",
    ToolFlags.PROJECTION: "glyph-projection",
    ToolFlags.PURIFICATION: "glyph-purification",
    ToolFlags.ANIMISMUS: "glyph-life-and-death",
    ToolFlags.QUINTESSENCE_GLYPHS: "glyph-unification",  # or glyph-dispersion
    ToolFlags.DUPLICATION: "glyph-duplication",
    ToolFlags.BONDER: "bonder",
    ToolFlags.UNBONDER: "unbonder",
    ToolFlags.MULTI_BONDER: "bonder-speed",
    ToolFlags.TRIPLEX_BONDER: "bonder-prisma",
}

VALID_ARM_TYPES = {t.value for t in ArmType}
VALID_GLYPH_TYPES = {t.value for t in GlyphType}
VALID_INSTRUCTIONS = set("GgRrEePpAaCXO")


@dataclass(frozen=True)
class StructureScore:
    """Result of structure scoring."""

    level: int  # 0-10
    score: float  # 0.0-1.0
    details: dict[str, bool]  # Which checks passed
    description: str  # Human-readable summary

    def __repr__(self) -> str:
        return f"StructureScore(level={self.level}, score={self.score:.2f}, desc='{self.description}')"


def score_solution_structure(
    solution_text: str,
    puzzle: Puzzle,
    omsim_error: str | None = None,
    omsim_error_cycle: int = -1,
    omsim_verified: bool = False,
) -> StructureScore:
    """Score a solution's structural correctness against a puzzle.

    Args:
        solution_text: Raw model output.
        puzzle: The puzzle being solved.
        omsim_error: Error message from omsim (if verification was attempted).
        omsim_error_cycle: Cycle at which omsim error occurred (-1 if N/A).
        omsim_verified: Whether omsim verified the solution as correct.

    Returns:
        StructureScore with level 0-10 and score 0.0-1.0.
    """
    details: dict[str, bool] = {}

    if omsim_verified:
        return StructureScore(
            level=10, score=1.0,
            details={"verified": True},
            description="Verified correct",
        )

    # ── Level 1: Has any valid solution lines? ──────────────────────
    lines = solution_text.strip().split("\n")
    solution_lines = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(k) for k in
               ["INPUT ", "OUTPUT ", "ARM ", "GLYPH ", "TRACK ", "TAPE:"]):
            solution_lines.append(stripped)

    details["has_solution_lines"] = len(solution_lines) > 0
    if not details["has_solution_lines"]:
        return StructureScore(
            level=0, score=0.0, details=details,
            description="No valid solution lines found",
        )

    # ── Level 2: Has INPUT and OUTPUT lines? ────────────────────────
    input_lines = [l for l in solution_lines if l.startswith("INPUT ")]
    output_lines = [l for l in solution_lines if l.startswith("OUTPUT ")]

    details["has_input"] = len(input_lines) > 0
    details["has_output"] = len(output_lines) > 0
    if not (details["has_input"] and details["has_output"]):
        missing = []
        if not details["has_input"]:
            missing.append("INPUT")
        if not details["has_output"]:
            missing.append("OUTPUT")
        return StructureScore(
            level=1, score=0.1, details=details,
            description=f"Missing {' and '.join(missing)} line(s)",
        )

    # ── Level 3: Correct number of I/O lines? ──────────────────────
    expected_inputs = len(puzzle.inputs)
    expected_outputs = len(puzzle.outputs)
    details["correct_input_count"] = len(input_lines) >= expected_inputs
    details["correct_output_count"] = len(output_lines) >= expected_outputs
    if not (details["correct_input_count"] and details["correct_output_count"]):
        return StructureScore(
            level=2, score=0.2, details=details,
            description=f"I/O count wrong: {len(input_lines)} inputs (need {expected_inputs}), "
                        f"{len(output_lines)} outputs (need {expected_outputs})",
        )

    # ── Level 4: Has ARM with TAPE? ─────────────────────────────────
    arm_lines = [l for l in solution_lines if l.startswith("ARM ")]
    tape_lines = [l for l in solution_lines if l.startswith("TAPE:") or l.lstrip().startswith("TAPE:")]
    details["has_arm"] = len(arm_lines) > 0
    details["has_tape"] = len(tape_lines) > 0
    if not (details["has_arm"] and details["has_tape"]):
        return StructureScore(
            level=3, score=0.3, details=details,
            description="Missing ARM or TAPE lines",
        )

    # ── Level 5: Has appropriate glyph(s)? ──────────────────────────
    glyph_lines = [l for l in solution_lines if l.startswith("GLYPH ")]
    glyph_types_used = set()
    for gl in glyph_lines:
        parts = gl.split()
        if len(parts) >= 2:
            glyph_types_used.add(parts[1])

    # Determine which glyphs this puzzle needs
    needed_glyphs: set[str] = set()
    for flag, glyph_name in _TOOL_TO_GLYPH.items():
        if puzzle.allowed_tools & flag:
            # Check if the puzzle actually requires this transformation
            # (it's in the allowed tools because the generator put it there)
            if flag != ToolFlags.BONDER:  # Bonder only needed if outputs have bonds
                needed_glyphs.add(glyph_name)

    # For bonding, only require if outputs have bonds
    if any(mol.bonds for mol in puzzle.outputs):
        if puzzle.allowed_tools & ToolFlags.BONDER:
            needed_glyphs.add("bonder")
        if puzzle.allowed_tools & ToolFlags.TRIPLEX_BONDER:
            needed_glyphs.add("bonder-prisma")

    # Remove disposal from "needed" — it's a convenience, not required
    needed_glyphs.discard("glyph-disposal")

    details["has_needed_glyphs"] = bool(
        needed_glyphs and glyph_types_used & (needed_glyphs | VALID_GLYPH_TYPES)
    ) or not needed_glyphs  # If no glyphs needed, pass

    if needed_glyphs and not glyph_types_used:
        return StructureScore(
            level=4, score=0.4, details=details,
            description=f"Missing GLYPH line(s), puzzle needs: {needed_glyphs}",
        )

    # ── Level 6: All type names valid? ──────────────────────────────
    invalid_arms = []
    for al in arm_lines:
        parts = al.split()
        if len(parts) >= 2 and parts[1] not in VALID_ARM_TYPES:
            invalid_arms.append(parts[1])

    invalid_glyphs = []
    for gl in glyph_lines:
        parts = gl.split()
        if len(parts) >= 2 and parts[1] not in VALID_GLYPH_TYPES:
            invalid_glyphs.append(parts[1])

    invalid_instructions = []
    for tl in tape_lines:
        # Extract instructions from TAPE: 1:G 2:R ...
        for match in re.finditer(r'\d+:([A-Za-z])', tl):
            instr = match.group(1)
            if instr not in VALID_INSTRUCTIONS:
                invalid_instructions.append(instr)

    details["valid_arm_types"] = len(invalid_arms) == 0
    details["valid_glyph_types"] = len(invalid_glyphs) == 0
    details["valid_instructions"] = len(invalid_instructions) == 0

    if invalid_arms or invalid_glyphs or invalid_instructions:
        problems = []
        if invalid_arms:
            problems.append(f"bad arm types: {invalid_arms}")
        if invalid_glyphs:
            problems.append(f"bad glyph types: {invalid_glyphs}")
        if invalid_instructions:
            problems.append(f"bad instructions: {set(invalid_instructions)}")
        return StructureScore(
            level=5, score=0.5, details=details,
            description=f"Invalid names: {'; '.join(problems)}",
        )

    # ── Levels 7-9: Require omsim feedback ──────────────────────────
    if omsim_error is None:
        # No omsim run yet — can't score higher
        return StructureScore(
            level=6, score=0.6, details=details,
            description="Structurally complete, not yet verified",
        )

    # Level 7: Check if it's a static overlap (cycle 0 error)
    details["no_static_overlap"] = "overlap" not in omsim_error.lower() or omsim_error_cycle > 0
    if not details["no_static_overlap"]:
        return StructureScore(
            level=6, score=0.6, details=details,
            description=f"Static overlap at cycle 0: {omsim_error[:80]}",
        )

    # Level 8: Did omsim load and start simulating?
    details["omsim_loaded"] = omsim_error_cycle >= 0 or "did not complete" in omsim_error
    if not details["omsim_loaded"]:
        return StructureScore(
            level=7, score=0.7, details=details,
            description=f"omsim load error: {omsim_error[:80]}",
        )

    # Level 9: Ran for some cycles before failing
    if omsim_error_cycle > 0:
        # Scale 0.8-0.9 based on how far it got (more cycles = closer to working)
        cycle_bonus = min(omsim_error_cycle / 100.0, 1.0) * 0.1
        return StructureScore(
            level=9, score=0.8 + cycle_bonus, details=details,
            description=f"Runtime error at cycle {omsim_error_cycle}: {omsim_error[:60]}",
        )

    # Ran but didn't complete (e.g. cycle limit)
    return StructureScore(
        level=8, score=0.8, details=details,
        description=f"Simulation failed: {omsim_error[:80]}",
    )
