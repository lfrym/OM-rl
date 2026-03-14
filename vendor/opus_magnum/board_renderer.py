"""Render a text-based board view showing placed parts and diagnostics.

This helps models understand the spatial layout of their solution and
debug collisions/errors.
"""

from __future__ import annotations

from .models import (
    Arm,
    ArmType,
    Glyph,
    GlyphType,
    HexVector,
    IOPart,
    IOType,
    Puzzle,
    Solution,
    Track,
    GLYPH_FOOTPRINTS,
    HEX_DIRECTIONS,
)


def render_solution_summary(
    solution: Solution,
    puzzle: Puzzle | None = None,
    error_msg: str | None = None,
    error_cycle: int = -1,
    error_location: tuple[int, int] | None = None,
) -> str:
    """Render a rich text summary of a solution's layout and any errors."""
    lines: list[str] = []

    # ── Part placement summary ──
    lines.append("SOLUTION LAYOUT:")
    lines.append("")

    # Track all occupied hexes for overlap detection
    # hex_pos -> list of (part_type, part_desc)
    occupied: dict[tuple[int, int], list[tuple[str, str]]] = {}

    def mark(pos: HexVector, part_type: str, desc: str):
        key = (pos.u, pos.v)
        occupied.setdefault(key, []).append((part_type, desc))

    # Arms
    if solution.arms:
        lines.append(f"ARMS ({len(solution.arms)}):")
        for arm in solution.arms:
            gripper = _gripper_position(arm)
            tape_summary = _summarize_tape(arm)
            lines.append(
                f"  [{arm.arm_id}] {arm.arm_type.value} at ({arm.position.u},{arm.position.v}) "
                f"rot={arm.rotation} ext={arm.extension} "
                f"-> gripper at ({gripper.u},{gripper.v})"
            )
            lines.append(f"       tape: {tape_summary}")
            mark(arm.position, "ARM_BASE", f"arm[{arm.arm_id}] base")
            mark(gripper, "ARM_GRIP", f"arm[{arm.arm_id}] gripper")
        lines.append("")

    # Glyphs
    if solution.glyphs:
        lines.append(f"GLYPHS ({len(solution.glyphs)}):")
        for glyph in solution.glyphs:
            footprint = _glyph_world_footprint(glyph)
            hex_strs = [f"({h.u},{h.v})" for h in footprint]
            lines.append(
                f"  {glyph.glyph_type.value} at ({glyph.position.u},{glyph.position.v}) "
                f"rot={glyph.rotation} -> occupies {' '.join(hex_strs)}"
            )
            for h in footprint:
                mark(h, "GLYPH", glyph.glyph_type.value)
        lines.append("")

    # Tracks
    if solution.tracks:
        lines.append(f"TRACKS ({len(solution.tracks)}):")
        for i, track in enumerate(solution.tracks):
            path_strs = [f"({p.u},{p.v})" for p in track.positions]
            lines.append(f"  track[{i}]: {' -> '.join(path_strs)}")
            for p in track.positions:
                mark(p, "TRACK", f"track[{i}]")
        lines.append("")

    # I/O
    if solution.ios:
        inputs = [io for io in solution.ios if io.io_type == IOType.INPUT]
        outputs = [io for io in solution.ios if io.io_type != IOType.INPUT]
        if inputs:
            lines.append(f"INPUTS ({len(inputs)}):")
            for io in inputs:
                lines.append(
                    f"  reagent[{io.molecule_index}] at ({io.position.u},{io.position.v}) rot={io.rotation}"
                )
                if puzzle and io.molecule_index < len(puzzle.inputs):
                    mol = puzzle.inputs[io.molecule_index]
                    for atom_pos, atom in mol.atoms:
                        world_pos = _rotate_and_translate(atom_pos, io.position, io.rotation)
                        mark(world_pos, "INPUT_ATOM", f"input[{io.molecule_index}] {atom.name}")
        if outputs:
            lines.append(f"OUTPUTS ({len(outputs)}):")
            for io in outputs:
                lines.append(
                    f"  product[{io.molecule_index}] at ({io.position.u},{io.position.v}) rot={io.rotation}"
                )
                if puzzle and io.molecule_index < len(puzzle.outputs):
                    mol = puzzle.outputs[io.molecule_index]
                    for atom_pos, atom in mol.atoms:
                        world_pos = _rotate_and_translate(atom_pos, io.position, io.rotation)
                        mark(world_pos, "OUTPUT_ATOM", f"output[{io.molecule_index}] {atom.name}")
        lines.append("")

    # ── Overlap detection (static) ──
    overlaps = {pos: parts for pos, parts in occupied.items() if len(parts) > 1}
    if overlaps:
        lines.append("STATIC OVERLAPS DETECTED:")
        for (u, v), parts in sorted(overlaps.items()):
            part_descs = [f"{ptype}({desc})" for ptype, desc in parts]
            lines.append(f"  ({u},{v}): {', '.join(part_descs)}")
        lines.append("")

    # ── Hex grid map ──
    lines.append("BOARD MAP:")
    lines.append(_render_hex_map(occupied, error_location))
    lines.append("")

    # ── Error info ──
    if error_msg:
        lines.append("ERROR:")
        lines.append(f"  {error_msg}")
        if error_cycle >= 0:
            lines.append(f"  At cycle: {error_cycle}")
        if error_location:
            eu, ev = error_location
            lines.append(f"  At location: ({eu},{ev})")
            # What's near the error?
            nearby = []
            for (pu, pv), parts in occupied.items():
                dist = _hex_distance(eu, ev, pu, pv)
                if dist <= 2:
                    for ptype, desc in parts:
                        nearby.append((dist, pu, pv, ptype, desc))
            if nearby:
                nearby.sort()
                lines.append("  Nearby parts:")
                for dist, pu, pv, ptype, desc in nearby:
                    lines.append(f"    dist={dist}: ({pu},{pv}) {ptype}({desc})")
        lines.append("")

    return "\n".join(lines)


def _gripper_position(arm: Arm) -> HexVector:
    """Calculate where the arm's gripper is based on position, rotation, extension."""
    direction = HEX_DIRECTIONS[arm.rotation % 6]
    return HexVector(
        arm.position.u + direction.u * arm.extension,
        arm.position.v + direction.v * arm.extension,
    )


def _glyph_world_footprint(glyph: Glyph) -> list[HexVector]:
    """Get the world-space hex positions a glyph occupies."""
    local_footprint = GLYPH_FOOTPRINTS.get(glyph.glyph_type, [HexVector(0, 0)])
    result = []
    for offset in local_footprint:
        rotated = offset.rotate(glyph.rotation)
        result.append(
            HexVector(
                glyph.position.u + rotated.u,
                glyph.position.v + rotated.v,
            )
        )
    return result


def _rotate_and_translate(offset: HexVector, origin: HexVector, rotation: int) -> HexVector:
    """Rotate offset by rotation steps, then translate to origin."""
    rotated = offset.rotate(rotation)
    return HexVector(origin.u + rotated.u, origin.v + rotated.v)


def _hex_distance(u1: int, v1: int, u2: int, v2: int) -> int:
    du = u1 - u2
    dv = v1 - v2
    return max(abs(du), abs(dv), abs(du + dv))


def _summarize_tape(arm: Arm) -> str:
    """Summarize an arm's instruction tape."""
    if not arm.instructions:
        return "(empty)"
    sorted_instrs = sorted(arm.instructions.items())
    min_c = sorted_instrs[0][0]
    max_c = sorted_instrs[-1][0]
    tape_str = " ".join(f"{c}:{i.value}" for c, i in sorted_instrs)
    has_repeat = any(i.value == "C" for i in arm.instructions.values())
    if has_repeat:
        tape_str += " (loops)"
    return f"cycles {min_c}-{max_c}: {tape_str}"


def _render_hex_map(
    occupied: dict[tuple[int, int], list[tuple[str, str]]],
    error_location: tuple[int, int] | None = None,
) -> str:
    """Render a simple text grid showing occupied hexes.

    Uses a compact 2-char representation per hex cell.
    """
    if not occupied and not error_location:
        return "  (empty board)"

    all_positions = set(occupied.keys())
    if error_location:
        all_positions.add(error_location)

    if not all_positions:
        return "  (empty board)"

    min_u = min(p[0] for p in all_positions) - 1
    max_u = max(p[0] for p in all_positions) + 1
    min_v = min(p[1] for p in all_positions) - 1
    max_v = max(p[1] for p in all_positions) + 1

    # Clamp to reasonable size
    if max_u - min_u > 20 or max_v - min_v > 20:
        return "  (board too large to render, see part positions above)"

    # Symbol mapping
    def cell_symbol(u: int, v: int) -> str:
        is_error = error_location and (u, v) == error_location
        parts = occupied.get((u, v), [])

        if not parts and not is_error:
            return " . "

        if is_error and not parts:
            return " X "

        types = {p[0] for p in parts}
        # Priority rendering
        if len(parts) > 1:
            sym = "!!" if not is_error else "!X"
        elif "ARM_BASE" in types:
            sym = "Ab"
        elif "ARM_GRIP" in types:
            sym = "Ag"
        elif "GLYPH" in types:
            sym = "Gl"
        elif "TRACK" in types:
            sym = "Tk"
        elif "INPUT_ATOM" in types:
            sym = "In"
        elif "OUTPUT_ATOM" in types:
            sym = "Ou"
        else:
            sym = "??"

        if is_error:
            return f"*{sym[0]}"
        return f" {sym}"

    lines = []
    # Header with u coordinates
    header = "  v\\u"
    for u in range(min_u, max_u + 1):
        header += f"{u:>3}"
    lines.append(header)

    for v in range(min_v, max_v + 1):
        indent = " " * (v - min_v)  # Hex stagger
        row = f"{indent}{v:>3} "
        for u in range(min_u, max_u + 1):
            row += cell_symbol(u, v)
        lines.append(row)

    lines.append("")
    lines.append("  Legend: Ab=arm base, Ag=arm gripper, Gl=glyph, Tk=track,")
    lines.append("          In=input atom, Ou=output atom, !!=overlap, X=error location")

    return "\n".join(lines)
