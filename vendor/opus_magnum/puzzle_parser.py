"""Parse binary .puzzle files into Python Puzzle objects."""

from __future__ import annotations

import struct
from pathlib import Path

from .models import (
    Atom,
    Bond,
    BondType,
    HexVector,
    Molecule,
    Puzzle,
    ToolFlags,
)


class _Reader:
    """Stateful reader over a bytes buffer."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def byte(self) -> int:
        val = self.data[self.pos]
        self.pos += 1
        return val

    def signed_byte(self) -> int:
        val = self.byte()
        return val if val <= 127 else val - 256

    def uint32(self) -> int:
        val = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return val

    def int32(self) -> int:
        val = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return val

    def uint64(self) -> int:
        val = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return val

    def string(self) -> str:
        length = 0
        shift = 0
        while True:
            b = self.byte()
            length |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        raw = self.data[self.pos : self.pos + length]
        self.pos += length
        return raw.decode("utf-8")


def _parse_molecule(r: _Reader) -> Molecule:
    num_atoms = r.uint32()
    atoms: list[tuple[HexVector, Atom]] = []
    for _ in range(num_atoms):
        atom_type = Atom(r.byte())
        u = r.signed_byte()
        v = r.signed_byte()
        atoms.append((HexVector(u, v), atom_type))

    num_bonds = r.uint32()
    bonds: list[Bond] = []
    for _ in range(num_bonds):
        bond_type = BondType(r.byte())
        fu, fv = r.signed_byte(), r.signed_byte()
        tu, tv = r.signed_byte(), r.signed_byte()
        bonds.append(Bond(HexVector(fu, fv), HexVector(tu, tv), bond_type))

    return Molecule(atoms=atoms, bonds=bonds)


def parse_puzzle(source: str | Path | bytes) -> Puzzle:
    """Parse a .puzzle file from a path or raw bytes."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = source

    r = _Reader(raw)

    version = r.uint32()
    if version != 3:
        raise ValueError(f"Unsupported puzzle format version: {version}")

    name = r.string()
    creator_id = r.uint64()
    parts_available = r.uint64()
    allowed_tools = ToolFlags(parts_available)

    num_inputs = r.uint32()
    inputs = [_parse_molecule(r) for _ in range(num_inputs)]

    num_outputs = r.uint32()
    outputs = [_parse_molecule(r) for _ in range(num_outputs)]

    output_scale = r.uint32()
    is_production = bool(r.byte())

    # Skip production-specific data for now.

    return Puzzle(
        name=name,
        creator_id=creator_id,
        allowed_tools=allowed_tools,
        inputs=inputs,
        outputs=outputs,
        output_scale=output_scale,
        is_production=is_production,
        raw_bytes=raw,
    )
