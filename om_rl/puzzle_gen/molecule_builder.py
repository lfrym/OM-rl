"""Random molecule construction on a hex grid.

Builds molecules by growing outward from an origin atom, adding atoms
at random neighboring positions and optionally creating bonds between them.
"""

from __future__ import annotations

import random

from vendor.opus_magnum.models import (
    Atom,
    Bond,
    BondType,
    HexVector,
    HEX_DIRECTIONS,
    Molecule,
)


def build_molecule(
    atoms: list[Atom],
    bond_probability: float = 0.0,
    triplex_probability: float = 0.0,
    rng: random.Random | None = None,
) -> Molecule:
    """Build a molecule by placing atoms on a hex grid growing from the origin.

    Args:
        atoms: List of atom types to place. First atom goes at (0,0).
        bond_probability: Probability of adding a normal bond between adjacent atoms.
            Set to 1.0 for fully-bonded molecules.
        triplex_probability: Probability that a bond is triplex instead of normal.
            Only applies when a bond is being created.
        rng: Random number generator. Uses default if None.

    Returns:
        A Molecule with all atoms placed and bonds created.
    """
    if not atoms:
        return Molecule(atoms=[], bonds=[])

    if rng is None:
        rng = random.Random()

    placed: list[tuple[HexVector, Atom]] = []
    occupied: set[tuple[int, int]] = set()
    bonds: list[Bond] = []

    # Place first atom at origin
    origin = HexVector(0, 0)
    placed.append((origin, atoms[0]))
    occupied.add((0, 0))

    # Place remaining atoms at random free neighbors of existing atoms
    for atom_type in atoms[1:]:
        # Find all free neighbor positions adjacent to any placed atom
        candidates: list[HexVector] = []
        for pos, _ in placed:
            for d in HEX_DIRECTIONS:
                neighbor = pos + d
                key = (neighbor.u, neighbor.v)
                if key not in occupied:
                    candidates.append(neighbor)

        if not candidates:
            # Shouldn't happen for reasonable molecule sizes, but handle gracefully
            # by expanding search radius
            raise ValueError(
                f"No free neighbor positions available after placing {len(placed)} atoms"
            )

        pos = rng.choice(candidates)
        placed.append((pos, atom_type))
        occupied.add((pos.u, pos.v))

    # Optionally add bonds between adjacent atoms
    if bond_probability > 0:
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                pos_i, _ = placed[i]
                pos_j, _ = placed[j]
                # Check if they're hex-adjacent (distance 1)
                if pos_i.hex_distance(pos_j) == 1:
                    if rng.random() < bond_probability:
                        if triplex_probability > 0 and rng.random() < triplex_probability:
                            bond_type = BondType.TRIPLEX
                        else:
                            bond_type = BondType.NORMAL
                        bonds.append(Bond(pos_i, pos_j, bond_type))

    return Molecule(atoms=placed, bonds=bonds)


def build_single_atom_molecule(atom_type: Atom) -> Molecule:
    """Create a simple single-atom molecule at the origin."""
    return Molecule(
        atoms=[(HexVector(0, 0), atom_type)],
        bonds=[],
    )


def build_linear_molecule(
    atoms: list[Atom],
    direction: int = 0,
    bond_type: BondType = BondType.NORMAL,
) -> Molecule:
    """Build a straight-line molecule along a hex direction.

    Args:
        atoms: Atom types in order.
        direction: Hex direction index (0=E, 1=SE, 2=SW, 3=W, 4=NW, 5=NE).
        bond_type: Bond type between adjacent atoms.
    """
    placed: list[tuple[HexVector, Atom]] = []
    bonds: list[Bond] = []
    d = HEX_DIRECTIONS[direction % 6]

    for i, atom_type in enumerate(atoms):
        pos = HexVector(d.u * i, d.v * i)
        placed.append((pos, atom_type))
        if i > 0:
            prev_pos = placed[i - 1][0]
            bonds.append(Bond(prev_pos, pos, bond_type))

    return Molecule(atoms=placed, bonds=bonds)
