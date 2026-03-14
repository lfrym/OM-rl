"""Serialize Puzzle objects to binary .puzzle format (v3).

This is the inverse of vendor/opus_magnum/puzzle_parser.py.
"""

from __future__ import annotations

import struct
from pathlib import Path

from vendor.opus_magnum.models import Bond, Molecule, Puzzle


class _Writer:
    """Stateful writer to a bytes buffer."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def byte(self, val: int) -> None:
        self._parts.append(struct.pack("B", val & 0xFF))

    def signed_byte(self, val: int) -> None:
        self._parts.append(struct.pack("b", val))

    def uint32(self, val: int) -> None:
        self._parts.append(struct.pack("<I", val))

    def int32(self, val: int) -> None:
        self._parts.append(struct.pack("<i", val))

    def uint64(self, val: int) -> None:
        self._parts.append(struct.pack("<Q", val))

    def string(self, val: str) -> None:
        encoded = val.encode("utf-8")
        length = len(encoded)
        # Write varint-encoded length
        while length > 0x7F:
            self._parts.append(struct.pack("B", (length & 0x7F) | 0x80))
            length >>= 7
        self._parts.append(struct.pack("B", length & 0x7F))
        self._parts.append(encoded)

    def to_bytes(self) -> bytes:
        return b"".join(self._parts)


def _write_molecule(w: _Writer, mol: Molecule) -> None:
    w.uint32(len(mol.atoms))
    for pos, atom_type in mol.atoms:
        w.byte(int(atom_type))
        w.signed_byte(pos.u)
        w.signed_byte(pos.v)

    w.uint32(len(mol.bonds))
    for bond in mol.bonds:
        w.byte(int(bond.bond_type))
        w.signed_byte(bond.from_pos.u)
        w.signed_byte(bond.from_pos.v)
        w.signed_byte(bond.to_pos.u)
        w.signed_byte(bond.to_pos.v)


def write_puzzle(puzzle: Puzzle) -> bytes:
    """Serialize a Puzzle to binary .puzzle format (version 3)."""
    w = _Writer()

    w.uint32(3)  # version
    w.string(puzzle.name)
    w.uint64(puzzle.creator_id)
    w.uint64(int(puzzle.allowed_tools))

    w.uint32(len(puzzle.inputs))
    for mol in puzzle.inputs:
        _write_molecule(w, mol)

    w.uint32(len(puzzle.outputs))
    for mol in puzzle.outputs:
        _write_molecule(w, mol)

    w.uint32(puzzle.output_scale)
    w.byte(1 if puzzle.is_production else 0)

    return w.to_bytes()


def save_puzzle(puzzle: Puzzle, path: str | Path) -> None:
    """Write a puzzle to a file and update its raw_bytes."""
    data = write_puzzle(puzzle)
    puzzle.raw_bytes = data
    Path(path).write_bytes(data)
