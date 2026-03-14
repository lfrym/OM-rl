"""Write Solution objects to binary .solution format (version 7)."""

from __future__ import annotations

import struct
from pathlib import Path

from .models import (
    ArmType,
    GlyphType,
    IOType,
    Solution,
)

_ARM_TYPES = {t.value for t in ArmType}
_GLYPH_TYPES = {t.value for t in GlyphType}
_IO_TYPES = {t.value for t in IOType}


class _Writer:
    """Stateful binary writer."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def byte(self, val: int) -> None:
        self.buf.append(val & 0xFF)

    def uint32(self, val: int) -> None:
        self.buf.extend(struct.pack("<I", val))

    def int32(self, val: int) -> None:
        self.buf.extend(struct.pack("<i", val))

    def string(self, val: str) -> None:
        encoded = val.encode("utf-8")
        length = len(encoded)
        # Variable-length integer encoding (same as .NET BinaryWriter).
        while length >= 0x80:
            self.byte((length & 0x7F) | 0x80)
            length >>= 7
        self.byte(length)
        self.buf.extend(encoded)

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


def write_solution(solution: Solution) -> bytes:
    """Serialize a Solution to the binary .solution format."""
    w = _Writer()

    # Version
    w.uint32(7)

    # Puzzle name and solution name
    w.string(solution.puzzle_name)
    w.string(solution.solution_name)

    # Unsolved (0 metrics) — omsim will calculate the real metrics.
    w.uint32(0)

    # Count total parts.
    total_parts = (
        len(solution.arms) + len(solution.glyphs) + len(solution.tracks) + len(solution.ios)
    )
    w.uint32(total_parts)

    # Write arms.
    for arm in solution.arms:
        _write_arm_part(w, arm)

    # Write glyphs.
    for glyph in solution.glyphs:
        _write_glyph_part(w, glyph)

    # Write tracks.
    for track in solution.tracks:
        _write_track_part(w, track)

    # Write I/O placements.
    for io in solution.ios:
        _write_io_part(w, io)

    return w.to_bytes()


def save_solution(solution: Solution, path: str | Path) -> None:
    """Write a solution to a file."""
    Path(path).write_bytes(write_solution(solution))


def _write_arm_part(w: _Writer, arm) -> None:
    w.string(arm.arm_type.value)
    w.byte(1)  # version
    w.int32(arm.position.u)
    w.int32(arm.position.v)
    w.uint32(arm.extension)
    w.int32(arm.rotation)
    w.uint32(0)  # which_input_or_output (N/A for arms)

    # Instructions
    sorted_instr = sorted(arm.instructions.items())
    w.uint32(len(sorted_instr))
    for cycle, instr in sorted_instr:
        w.int32(cycle)
        w.byte(ord(instr.value))

    # No track data for arms.
    # Arm number
    w.uint32(arm.arm_id)


def _write_glyph_part(w: _Writer, glyph) -> None:
    w.string(glyph.glyph_type.value)
    w.byte(1)
    w.int32(glyph.position.u)
    w.int32(glyph.position.v)
    w.uint32(1)  # size (always 1 for glyphs)
    w.int32(glyph.rotation)
    w.uint32(0)  # which_input_or_output

    w.uint32(0)  # no instructions
    w.uint32(0)  # arm_number (0 for non-arms)


def _write_track_part(w: _Writer, track) -> None:
    w.string("track")
    w.byte(1)
    # Position: first hex of the track path.
    w.int32(track.positions[0].u if track.positions else 0)
    w.int32(track.positions[0].v if track.positions else 0)
    w.uint32(1)  # size
    w.int32(0)  # rotation
    w.uint32(0)  # which_input_or_output

    w.uint32(0)  # no instructions

    # Track hex positions (relative offsets from the part position).
    w.uint32(len(track.positions))
    for pos in track.positions:
        w.int32(pos.u)
        w.int32(pos.v)

    w.uint32(0)  # arm_number


def _write_io_part(w: _Writer, io) -> None:
    w.string(io.io_type.value)
    w.byte(1)
    w.int32(io.position.u)
    w.int32(io.position.v)
    w.uint32(1)  # size
    w.int32(io.rotation)
    w.uint32(io.molecule_index)

    w.uint32(0)  # no instructions
    w.uint32(0)  # arm_number
