"""Format puzzle state as text observations for the model.

Adapts the benchmark prompt templates for the RL environment,
with additions for iterative feedback and simulation state.
"""

from __future__ import annotations

from vendor.opus_magnum.models import Puzzle
from vendor.opus_magnum.text_format import puzzle_to_text


# Condensed game reference (shorter than benchmark version to save tokens)
GAME_REFERENCE = """\
COORDINATE SYSTEM: Axial hex (u,v). Directions: E=(+1,0) SE=(0,+1) SW=(-1,+1) W=(-1,0) NW=(0,-1) NE=(+1,-1)
Rotation 0=East. Each step is 60° CW. Gripper = base + direction[rot%6] * extension.

ARM MECHANICS: G=grab, g=drop, R/r=rotateCW/CCW, E/e=extend/retract, P/p=pivotCW/CCW, A/a=forward/back on track.
Arms rotate around their base. Gripper carries held atoms during rotation.

I/O: Inputs spawn reagents at their position (respawn when picked up). Outputs check for correct molecule shape+rotation.
Must produce 6 * output_scale copies of each product.

CYCLE: Half 1: grab/drop. Half 2: movement. After each: inputs spawn, glyphs activate, outputs checked.
Collisions (overlapping atoms/parts) halt the machine.

GLYPHS activate on ungrabbed atoms on their tiles:
- Bonding (2 hex): creates normal bond. Unbonding (2 hex): breaks bond.
- Calcification (1 hex): cardinal->salt. Projection (2 hex): quicksilver+metal->next metal.
- Purification (3 hex): 2x same metal->next metal. Animismus (4 hex): salt->vitae+mors.
- Unification (5 hex): air+fire+water+earth->quintessence. Dispersion (5 hex): reverse.
- Disposal (7 hex): destroys atoms.

OVERLAP RULES: No two parts may share a hex at cycle 0 (arm bases, grippers, glyph tiles, track hexes, I/O atoms).
"""

SOLUTION_FORMAT = """\
SOLUTION FORMAT:
INPUT pos=(<u>,<v>) rot=<0-5> idx=<molecule_index>
OUTPUT pos=(<u>,<v>) rot=<0-5> idx=<molecule_index>
ARM <type> pos=(<u>,<v>) rot=<0-5> ext=<length> id=<n>
  TAPE: <cycle>:<instruction> <cycle>:<instruction> ...
GLYPH <type> pos=(<u>,<v>) rot=<0-5>
TRACK pos=(<u1>,<v1>) (<u2>,<v2>) ...

Arm types: arm1, arm2, arm3, arm6, piston, baron
Glyph types: bonder, unbonder, bonder-speed, bonder-prisma, glyph-calcification, \
glyph-duplication, glyph-projection, glyph-purification, glyph-life-and-death, \
glyph-disposal, glyph-marker, glyph-unification, glyph-dispersion
Instructions: G=grab g=drop R=rotateCW r=rotateCCW E=extend e=retract P=pivotCW p=pivotCCW A=forward a=back C=repeat X=reset O=noop
"""


def format_initial_observation(puzzle: Puzzle) -> str:
    """Format the initial observation for a puzzle (no prior attempts)."""
    puzzle_text = puzzle_to_text(puzzle)
    io_text = _format_io_requirements(puzzle)

    return f"""\
Solve this Opus Magnum puzzle. Output a complete solution.

{GAME_REFERENCE}
{SOLUTION_FORMAT}
{puzzle_text}
{io_text}
Output ONLY the solution — no explanation needed."""


def format_feedback_observation(
    puzzle: Puzzle,
    error_message: str,
    attempt_num: int,
    max_attempts: int,
    board_state: str | None = None,
) -> str:
    """Format observation after a failed attempt, including error feedback."""
    lines = [
        f"Attempt {attempt_num}/{max_attempts} failed.",
        f"Error: {error_message}",
    ]
    if board_state:
        lines.append(f"\nBoard state at error:\n{board_state}")
    lines.append("\nFix the issue and submit a new solution.")
    return "\n".join(lines)


def _format_io_requirements(puzzle: Puzzle) -> str:
    """Format I/O count requirements."""
    num_inputs = len(puzzle.inputs)
    num_outputs = len(puzzle.outputs)
    input_indices = ", ".join(str(i) for i in range(num_inputs))
    output_indices = ", ".join(str(i) for i in range(num_outputs))

    return (
        f"I/O PLACEMENT:\n"
        f"Place inputs and outputs anywhere on the board.\n"
        f"  Required inputs: {num_inputs} (molecule indices: {input_indices})\n"
        f"  Required outputs: {num_outputs} (molecule indices: {output_indices})\n"
        f"Arms grab atoms FROM inputs and deliver assembled molecules TO outputs."
    )
