"""Read binary .solution files (version 7) and extract parts.

This is the inverse of solution_writer.py — used to extract canonical I/O
positions from reference (leaderboard) solutions.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .models import (
    Arm,
    ArmType,
    Glyph,
    GlyphType,
    HexVector,
    Instruction,
    IOPart,
    IOType,
    Solution,
    Track,
)

_ARM_TYPE_NAMES = {t.value for t in ArmType}
_GLYPH_TYPE_NAMES = {t.value for t in GlyphType}
_IO_TYPE_NAMES = {t.value for t in IOType}
_INSTR_BY_BYTE = {ord(i.value): i for i in Instruction}


class _Reader:
    """Stateful binary reader."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def byte(self) -> int:
        val = self.data[self.pos]
        self.pos += 1
        return val

    def uint32(self) -> int:
        val = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return val

    def int32(self) -> int:
        val = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return val

    def string(self) -> str:
        """Read a .NET-style length-prefixed string."""
        length = 0
        shift = 0
        while True:
            b = self.byte()
            length |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        raw = self.data[self.pos : self.pos + length]
        self.pos += length
        return raw.decode("utf-8")


def read_solution_ios(data: bytes) -> list[IOPart]:
    """Extract I/O placements from a binary .solution file."""
    r = _Reader(data)

    version = r.uint32()
    if version != 7:
        raise ValueError(f"Unsupported solution version: {version}")

    _puzzle_name = r.string()
    _solution_name = r.string()

    # Solved flag + optional metrics
    solved_flag = r.uint32()
    if solved_flag:
        # 4 metric pairs: (uint32 id, uint32 value)
        for _ in range(4):
            r.uint32()  # metric id
            r.uint32()  # metric value

    num_parts = r.uint32()
    ios: list[IOPart] = []

    for _ in range(num_parts):
        part_name = r.string()
        _version_byte = r.byte()  # always 1
        pos_u = r.int32()
        pos_v = r.int32()
        _size = r.uint32()
        rotation = r.int32()
        which_io = r.uint32()

        num_instructions = r.uint32()
        for _ in range(num_instructions):
            r.int32()  # index
            r.byte()  # instruction

        if part_name == "track":
            num_hexes = r.uint32()
            for _ in range(num_hexes):
                r.int32()
                r.int32()

        _arm_number = r.uint32()

        if part_name == "pipe":
            _conduit_id = r.uint32()
            num_conduit_hexes = r.uint32()
            for _ in range(num_conduit_hexes):
                r.int32()
                r.int32()

        # Check if this is an I/O part
        if part_name in _IO_TYPE_NAMES:
            ios.append(
                IOPart(
                    io_type=IOType(part_name),
                    position=HexVector(pos_u, pos_v),
                    rotation=rotation,
                    molecule_index=which_io,
                )
            )

    return ios


def read_full_solution(data: bytes) -> Solution:
    """Read all parts from a binary .solution file into a Solution object."""
    r = _Reader(data)

    version = r.uint32()
    if version != 7:
        raise ValueError(f"Unsupported solution version: {version}")

    puzzle_name = r.string()
    solution_name = r.string()

    solved_flag = r.uint32()
    if solved_flag:
        for _ in range(4):
            r.uint32()
            r.uint32()

    num_parts = r.uint32()
    arms: list[Arm] = []
    glyphs: list[Glyph] = []
    tracks: list[Track] = []
    ios: list[IOPart] = []

    for _ in range(num_parts):
        part_name = r.string()
        _version_byte = r.byte()
        pos_u = r.int32()
        pos_v = r.int32()
        size = r.uint32()
        rotation = r.int32()
        which_io = r.uint32()

        num_instructions = r.uint32()
        instructions: dict[int, Instruction] = {}
        for _ in range(num_instructions):
            cycle = r.int32()
            instr_byte = r.byte()
            if instr_byte in _INSTR_BY_BYTE:
                instructions[cycle] = _INSTR_BY_BYTE[instr_byte]

        track_positions: list[HexVector] = []
        if part_name == "track":
            num_hexes = r.uint32()
            for _ in range(num_hexes):
                tu = r.int32()
                tv = r.int32()
                track_positions.append(HexVector(tu, tv))

        arm_number = r.uint32()

        if part_name == "pipe":
            _conduit_id = r.uint32()
            num_conduit_hexes = r.uint32()
            for _ in range(num_conduit_hexes):
                r.int32()
                r.int32()

        pos = HexVector(pos_u, pos_v)

        if part_name in _ARM_TYPE_NAMES:
            arms.append(
                Arm(
                    arm_type=ArmType(part_name),
                    position=pos,
                    rotation=rotation,
                    extension=size,
                    instructions=instructions,
                    arm_id=arm_number,
                )
            )
        elif part_name in _GLYPH_TYPE_NAMES:
            glyphs.append(
                Glyph(
                    glyph_type=GlyphType(part_name),
                    position=pos,
                    rotation=rotation,
                )
            )
        elif part_name == "track":
            tracks.append(Track(positions=track_positions))
        elif part_name in _IO_TYPE_NAMES:
            ios.append(
                IOPart(
                    io_type=IOType(part_name),
                    position=pos,
                    rotation=rotation,
                    molecule_index=which_io,
                )
            )

    return Solution(
        puzzle_name=puzzle_name,
        solution_name=solution_name,
        arms=arms,
        glyphs=glyphs,
        tracks=tracks,
        ios=ios,
    )


def read_solution_ios_from_file(path: str | Path) -> list[IOPart]:
    """Read I/O parts from a .solution file on disk."""
    return read_solution_ios(Path(path).read_bytes())
