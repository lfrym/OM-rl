"""Format puzzle state as text observations for the model.

Adapts the benchmark prompt templates for the RL environment,
with additions for iterative feedback and simulation state.
"""

from __future__ import annotations

from vendor.opus_magnum.board_renderer import render_solution_summary
from vendor.opus_magnum.models import Puzzle, Solution
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

PLACEMENT RULES: Each hex may hold at most one part. Arm bases and grippers cannot share a hex with glyphs, inputs, outputs, or other arms. Arms may be placed on track hexes.
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


FEW_SHOT_EXAMPLES = """\
WORKED EXAMPLE 1 — Simple pass-through (move Water from input to output):

Puzzle: 1 input (Water), 1 output (Water). No glyph needed, just transport.
One arm at (0,0) with ext=2 reaches input at (0,2) and output at (0,-2).
Gripper calculation: arm at (0,0) rot=1(SE) ext=2 -> gripper at (0,2)=input.
After 3x R (rot 1->2->3->4=NW), gripper at (0,-2)=output.

INPUT pos=(0,2) rot=0 idx=0
OUTPUT pos=(0,-2) rot=0 idx=0
ARM arm1 pos=(0,0) rot=1 ext=2 id=0
  TAPE: 1:G 2:R 3:R 4:R 5:g 6:R 7:R 8:R 9:C

WORKED EXAMPLE 2 — Converting Fire to Salt using glyph-calcification:

Puzzle: 1 input (Fire), 1 output (Salt). Requires glyph-calcification (converts cardinal->salt).
Vertical layout: input at (0,2), glyph at (0,0), output at (0,-2).
Two arms shuttle atoms: arm 0 moves input->glyph, arm 1 moves glyph->output.

INPUT pos=(0,2) rot=0 idx=0
OUTPUT pos=(0,-2) rot=0 idx=0
GLYPH glyph-calcification pos=(0,0) rot=0
ARM arm1 pos=(0,1) rot=1 ext=1 id=0
  TAPE: 1:G 2:R 3:R 4:R 5:g 6:R 7:R 8:R 9:C
ARM arm1 pos=(0,-1) rot=1 ext=1 id=1
  TAPE: 7:G 8:R 9:R 10:R 11:g 12:R 13:R 14:R 15:C

Key patterns:
- Compute gripper position: base + HEX_DIRECTIONS[rot % 6] * extension.
  HEX_DIRECTIONS: 0=E(+1,0), 1=SE(0,+1), 2=SW(-1,+1), 3=W(-1,0), 4=NW(0,-1), 5=NE(+1,-1)
- 3x R (rotate CW) moves the gripper 180° around the arm base.
- G=grab, g=drop (lowercase!), C=repeat from start of tape.
- Include a GLYPH line when atoms need transformation. Use exact names: glyph-calcification, glyph-projection, etc.
- Arm type is "arm1" (not "ARM 1" or just "1").
- Every puzzle is different — adapt positions and layout to the specific inputs/outputs required.
"""


def format_initial_observation(puzzle: Puzzle) -> str:
    """Format the initial observation for a puzzle (no prior attempts)."""
    puzzle_text = puzzle_to_text(puzzle)
    io_text = _format_io_requirements(puzzle)

    return f"""\
Solve this Opus Magnum puzzle. You may submit multiple attempts — \
each failed attempt will return the simulator error so you can iterate.

{GAME_REFERENCE}
{SOLUTION_FORMAT}
{FEW_SHOT_EXAMPLES}
━━━ YOUR PUZZLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{puzzle_text}
{io_text}
Output ONLY the solution — no preamble or explanation."""


def format_feedback_observation(
    solution: Solution | None,
    puzzle: Puzzle,
    error_message: str,
    error_cycle: int,
    error_location: tuple[int, int] | None,
    attempt_num: int,
    max_attempts: int,
) -> str:
    """Format observation after a failed attempt, including rich board diagnostics."""
    header = f"Attempt {attempt_num}/{max_attempts} failed.\n"

    if solution is None:
        # Parsing failed — no board to render
        hint = (
            "Your response could not be parsed. "
            "Check that every line matches the exact format:\n"
            "  INPUT pos=(<u>,<v>) rot=<0-5> idx=<n>\n"
            "  OUTPUT pos=(<u>,<v>) rot=<0-5> idx=<n>\n"
            "  ARM <type> pos=(<u>,<v>) rot=<0-5> ext=<n> id=<n>\n"
            "    TAPE: <cycle>:<instr> ...\n"
        )
        return f"{header}Parse error: {error_message}\n\n{hint}Submit a corrected solution."

    board = render_solution_summary(
        solution, puzzle,
        error_msg=error_message,
        error_cycle=error_cycle,
        error_location=error_location,
    )
    return f"{header}\n{board}\nFix the issue and submit a new solution."


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
