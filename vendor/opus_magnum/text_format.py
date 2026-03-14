"""Convert puzzles to/from text representations for model consumption.

Also parses model-produced text solutions back into Solution objects.
"""

from __future__ import annotations

from .models import (
    Arm,
    ArmType,
    Atom,
    BondType,
    Glyph,
    GlyphType,
    HexVector,
    IOPart,
    IOType,
    Instruction,
    Molecule,
    Puzzle,
    Solution,
    Track,
    ToolFlags,
    GLYPH_COSTS,
    ARM_COSTS,
    TRACK_COST_PER_HEX,
    METAL_ORDER,
    GLYPH_FOOTPRINTS,
)


# ── Puzzle -> Text ──────────────────────────────────────────────────────────


def puzzle_to_text(puzzle: Puzzle) -> str:
    """Render a puzzle as a human/model-readable text description."""
    lines: list[str] = []
    lines.append(f"PUZZLE: {puzzle.name}")
    lines.append("")

    # Available tools.
    tools = _describe_tools(puzzle.allowed_tools)
    lines.append("AVAILABLE TOOLS:")
    for t in tools:
        lines.append(f"  - {t}")
    lines.append("")

    # Inputs (reagents).
    lines.append(f"INPUTS ({len(puzzle.inputs)} reagent(s)):")
    for i, mol in enumerate(puzzle.inputs):
        lines.append(f"  Reagent {i}:")
        lines.extend(_describe_molecule(mol, indent=4))
    lines.append("")

    # Outputs (products).
    lines.append(f"OUTPUTS ({len(puzzle.outputs)} product(s)):")
    for i, mol in enumerate(puzzle.outputs):
        lines.append(f"  Product {i}:")
        lines.extend(_describe_molecule(mol, indent=4))
    lines.append("")

    lines.append(f"OUTPUT SCALE: {puzzle.output_scale}")
    lines.append(f"TARGET OUTPUTS: {puzzle.output_scale * 6}")

    return "\n".join(lines)


def _describe_molecule(mol: Molecule, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines = []
    lines.append(f"{pad}Atoms:")
    for pos, atom in mol.atoms:
        lines.append(f"{pad}  ({pos.u:+d}, {pos.v:+d}) {atom.name} [{atom.symbol}]")
    if mol.bonds:
        lines.append(f"{pad}Bonds:")
        for bond in mol.bonds:
            btype = _bond_type_str(bond.bond_type)
            lines.append(
                f"{pad}  ({bond.from_pos.u:+d}, {bond.from_pos.v:+d}) -- "
                f"({bond.to_pos.u:+d}, {bond.to_pos.v:+d}) [{btype}]"
            )
    return lines


def _bond_type_str(bt: BondType) -> str:
    if bt == BondType.NORMAL:
        return "normal"
    parts = []
    if bt & BondType.TRIPLEX_RED:
        parts.append("red")
    if bt & BondType.TRIPLEX_BLACK:
        parts.append("black")
    if bt & BondType.TRIPLEX_YELLOW:
        parts.append("yellow")
    return "triplex:" + "+".join(parts) if parts else f"bond({int(bt)})"


def _describe_tools(flags: ToolFlags) -> list[str]:
    tools = []
    # Arms
    arm_descriptions = [
        (ToolFlags.ARM, "Arm (single-grip, lengths 1-3) — 20g/30g/30g"),
        (ToolFlags.MULTI_ARM, "Multi-arm (6-grip hex arm) — 30g"),
        (ToolFlags.PISTON, "Piston (extendable arm) — 40g"),
        (ToolFlags.TRACK, "Track — 5g per hex"),
        (ToolFlags.VAN_BERLOS_WHEEL, "Van Berlo's Wheel — 30g"),
    ]
    for flag, desc in arm_descriptions:
        if flags & flag:
            tools.append(desc)

    glyph_descriptions = [
        (ToolFlags.BONDER, "Glyph of Bonding — 10g (2 hex, creates normal bond)"),
        (ToolFlags.UNBONDER, "Glyph of Unbonding — 10g (2 hex, breaks normal bond)"),
        (ToolFlags.MULTI_BONDER, "Glyph of Multi-Bonding — 30g (4 hex, bonds all adjacent pairs)"),
        (ToolFlags.TRIPLEX_BONDER, "Glyph of Triplex Bonding — 20g (3 hex, creates triplex bond)"),
        (ToolFlags.CALCIFICATION, "Glyph of Calcification — 10g (1 hex, cardinal -> salt)"),
        (ToolFlags.DUPLICATION, "Glyph of Duplication — 20g (2 hex, salt -> cardinal copy)"),
        (
            ToolFlags.PROJECTION,
            "Glyph of Projection — 20g (2 hex, quicksilver + metal -> next metal)",
        ),
        (ToolFlags.PURIFICATION, "Glyph of Purification — 20g (3 hex, 2x metal -> next metal)"),
        (ToolFlags.ANIMISMUS, "Glyph of Animismus — 20g (4 hex, 2 salt -> vitae + mors)"),
        (ToolFlags.DISPOSAL, "Glyph of Disposal — 0g (7 hex, destroys atoms)"),
        (
            ToolFlags.QUINTESSENCE_GLYPHS,
            "Glyph of Unification — 20g (5 hex, 4 cardinals -> quintessence)",
        ),
        (
            ToolFlags.QUINTESSENCE_GLYPHS,
            "Glyph of Dispersion — 20g (5 hex, quintessence -> 4 cardinals)",
        ),
    ]
    for flag, desc in glyph_descriptions:
        if flags & flag:
            tools.append(desc)

    # Instructions
    instr_descriptions = [
        (ToolFlags.GRAB_AND_TURN, "Instructions: Grab, Drop, Rotate CW/CCW"),
        (ToolFlags.PIVOT, "Instructions: Pivot CW/CCW"),
        (ToolFlags.RESET, "Instructions: Reset (return arm to start of track)"),
        (ToolFlags.REPEAT, "Instructions: Repeat (loop instruction tape)"),
    ]
    for flag, desc in instr_descriptions:
        if flags & flag:
            tools.append(desc)

    return tools


# ── Solution -> Text ───────────────────────────────────────────────────────


def solution_to_text(solution: Solution) -> str:
    """Convert a Solution object to the text format that models produce."""
    lines: list[str] = []

    for io in solution.ios:
        label = "INPUT" if io.io_type == IOType.INPUT else "OUTPUT"
        lines.append(
            f"{label} pos=({io.position.u},{io.position.v}) "
            f"rot={io.rotation} idx={io.molecule_index}"
        )

    for arm in solution.arms:
        line = (
            f"ARM {arm.arm_type.value} "
            f"pos=({arm.position.u},{arm.position.v}) "
            f"rot={arm.rotation} ext={arm.extension} id={arm.arm_id}"
        )
        lines.append(line)
        if arm.instructions:
            tape_entries = [
                f"{cycle}:{instr.value}" for cycle, instr in sorted(arm.instructions.items())
            ]
            lines.append(f"  TAPE: {' '.join(tape_entries)}")

    for glyph in solution.glyphs:
        lines.append(
            f"GLYPH {glyph.glyph_type.value} "
            f"pos=({glyph.position.u},{glyph.position.v}) rot={glyph.rotation}"
        )

    for track in solution.tracks:
        pos_strs = [f"({p.u},{p.v})" for p in track.positions]
        lines.append(f"TRACK pos={' '.join(pos_strs)}")

    return "\n".join(lines)


# ── Text Solution -> Solution object ────────────────────────────────────────

# The text format a model should produce:
SOLUTION_FORMAT_SPEC = """\
SOLUTION FORMAT:
A solution is a list of parts. Each part is defined on one or more lines.
You must place inputs, outputs, arms, glyphs, and tracks.

INPUT pos=(<u>,<v>) rot=<0-5> idx=<molecule_index>
OUTPUT pos=(<u>,<v>) rot=<0-5> idx=<molecule_index>

ARM <type> pos=(<u>,<v>) rot=<0-5> ext=<length> id=<n>
  TAPE: <cycle>:<instruction> <cycle>:<instruction> ...
  (Instructions: G=grab, g=drop, R=rotateCW, r=rotateCCW, E=extend, e=retract,
   P=pivotCW, p=pivotCCW, A=forward, a=back, C=repeat, X=reset, O=noop)

GLYPH <type> pos=(<u>,<v>) rot=<0-5>

TRACK pos=(<u1>,<v1>) (<u2>,<v2>) (<u3>,<v3>) ...

Arm types: arm1, arm2, arm3, arm6, piston, baron
Glyph types: bonder, unbonder, bonder-speed, bonder-prisma,
  glyph-calcification, glyph-duplication, glyph-projection,
  glyph-purification, glyph-life-and-death, glyph-disposal,
  glyph-marker, glyph-unification, glyph-dispersion

Hex coordinates use axial (u, v) system. Rotation 0-5 = multiples of 60° CW.
"""


def parse_text_solution(text: str, puzzle_name: str) -> Solution:
    """Parse a model-produced text solution into a Solution object."""
    arms: list[Arm] = []
    glyphs: list[Glyph] = []
    tracks: list[Track] = []
    ios: list[IOPart] = []

    lines = text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#") or line.startswith("SOLUTION"):
            continue

        if line.startswith("ARM "):
            arm, i = _parse_arm_line(line, lines, i)
            arms.append(arm)
        elif line.startswith("GLYPH "):
            glyphs.append(_parse_glyph_line(line))
        elif line.startswith("TRACK "):
            tracks.append(_parse_track_line(line))
        elif line.startswith("INPUT ") or line.startswith("OUTPUT "):
            ios.append(_parse_io_line(line))

    return Solution(
        puzzle_name=puzzle_name,
        solution_name="ai-solution",
        arms=arms,
        glyphs=glyphs,
        tracks=tracks,
        ios=ios,
    )


def _parse_pos(s: str) -> HexVector:
    """Parse '(<u>,<v>)' or '<u>,<v>'."""
    s = s.strip().strip("()")
    parts = s.split(",")
    return HexVector(int(parts[0].strip()), int(parts[1].strip()))


def _parse_kv(tokens: list[str]) -> dict[str, str]:
    """Parse key=value tokens."""
    result = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _parse_arm_line(line: str, lines: list[str], next_i: int) -> tuple[Arm, int]:
    parts = line.split()
    arm_type = ArmType(parts[1])
    kv = _parse_kv(parts[2:])
    pos = _parse_pos(kv["pos"])
    rot = int(kv.get("rot", "0"))
    ext = int(kv.get("ext", "1"))
    arm_id = int(kv.get("id", "0"))

    instructions: dict[int, Instruction] = {}
    # Check if next line is a TAPE line.
    while next_i < len(lines) and lines[next_i].strip().startswith("TAPE:"):
        tape_str = lines[next_i].strip().removeprefix("TAPE:").strip()
        for entry in tape_str.split():
            cycle_str, instr_char = entry.split(":")
            instructions[int(cycle_str)] = Instruction(instr_char)
        next_i += 1

    return (
        Arm(
            arm_type=arm_type,
            position=pos,
            rotation=rot,
            extension=ext,
            instructions=instructions,
            arm_id=arm_id,
        ),
        next_i,
    )


def _parse_glyph_line(line: str) -> Glyph:
    parts = line.split()
    glyph_type = GlyphType(parts[1])
    kv = _parse_kv(parts[2:])
    pos = _parse_pos(kv["pos"])
    rot = int(kv.get("rot", "0"))
    return Glyph(glyph_type=glyph_type, position=pos, rotation=rot)


def _parse_track_line(line: str) -> Track:
    # TRACK pos=(u1,v1) (u2,v2) (u3,v3) ...
    rest = line.removeprefix("TRACK").strip()
    if rest.startswith("pos="):
        rest = rest.removeprefix("pos=")
    positions = []
    import re

    for match in re.finditer(r"\(([^)]+)\)", rest):
        positions.append(_parse_pos(match.group(1)))
    return Track(positions=positions)


def _parse_io_line(line: str) -> IOPart:
    parts = line.split()
    if parts[0] == "INPUT":
        io_type = IOType.INPUT
    elif parts[0] == "OUTPUT":
        io_type = IOType.OUTPUT
    else:
        io_type = IOType.INFINITE
    kv = _parse_kv(parts[1:])
    pos = _parse_pos(kv["pos"])
    rot = int(kv.get("rot", "0"))
    idx = int(kv.get("idx", "0"))
    return IOPart(io_type=io_type, position=pos, rotation=rot, molecule_index=idx)
