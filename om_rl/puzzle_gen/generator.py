"""Puzzle generator: create random, valid Opus Magnum puzzles of controllable complexity.

The generator works backward from desired outputs:
1. Sample output molecule(s) with random atoms and structure
2. Determine which transformations are needed
3. Derive the required input atoms
4. Build input molecules
5. Set appropriate tool flags
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from vendor.opus_magnum.models import (
    Atom,
    BondType,
    GlyphType,
    HexVector,
    Molecule,
    Puzzle,
    ToolFlags,
)

from .chemistry import (
    ALL_TRANSFORMATIONS,
    CARDINALS,
    GLYPH_TO_TOOL,
    METAL_CHAIN,
    Transformation,
    derive_inputs_for_output,
    required_tools_for_path,
)
from .molecule_builder import build_molecule, build_single_atom_molecule
from .puzzle_writer import write_puzzle


# Base tools always available in generated puzzles
BASE_TOOLS = (
    ToolFlags.ARM
    | ToolFlags.MULTI_ARM
    | ToolFlags.PISTON
    | ToolFlags.TRACK
    | ToolFlags.GRAB_AND_TURN
    | ToolFlags.DROP
    | ToolFlags.RESET
    | ToolFlags.REPEAT
    | ToolFlags.PIVOT
)


@dataclass
class PuzzleSpec:
    """Specification for a generated puzzle."""

    complexity_level: int  # 1-5
    output_atoms: list[Atom]  # Atoms in output molecule(s)
    output_bonds: bool  # Whether output has bonds
    triplex_bonds: bool  # Whether to use triplex bonds
    num_outputs: int  # Number of distinct output molecules
    transformations: list[Transformation]  # Required glyph transformations
    input_atoms: list[Atom]  # Derived input atoms


# Atom pools by complexity level
LEVEL_ATOMS: dict[int, list[Atom]] = {
    1: [Atom.SALT, Atom.AIR, Atom.EARTH, Atom.FIRE, Atom.WATER],
    2: [Atom.SALT, Atom.AIR, Atom.EARTH, Atom.FIRE, Atom.WATER,
        Atom.QUICKSILVER, Atom.LEAD, Atom.TIN, Atom.IRON],
    3: [Atom.SALT, Atom.AIR, Atom.EARTH, Atom.FIRE, Atom.WATER,
        Atom.QUICKSILVER, Atom.LEAD, Atom.TIN, Atom.IRON, Atom.COPPER],
    4: [Atom.SALT, Atom.AIR, Atom.EARTH, Atom.FIRE, Atom.WATER,
        Atom.QUICKSILVER, Atom.LEAD, Atom.TIN, Atom.IRON, Atom.COPPER,
        Atom.SILVER, Atom.VITAE, Atom.MORS],
    5: [Atom.SALT, Atom.AIR, Atom.EARTH, Atom.FIRE, Atom.WATER,
        Atom.QUICKSILVER, Atom.LEAD, Atom.TIN, Atom.IRON, Atom.COPPER,
        Atom.SILVER, Atom.GOLD, Atom.VITAE, Atom.MORS, Atom.QUINTESSENCE],
}

# Atom count ranges by level
LEVEL_ATOM_COUNTS: dict[int, tuple[int, int]] = {
    1: (1, 2),
    2: (2, 4),
    3: (3, 6),
    4: (4, 8),
    5: (6, 12),
}

# Number of output molecules by level
LEVEL_OUTPUT_COUNTS: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (1, 1),
    3: (1, 1),
    4: (1, 2),
    5: (1, 3),
}


# Atoms that REQUIRE a glyph transformation (not available as direct puzzle inputs)
# Used to ensure puzzles aren't trivial pass-throughs.
_NEEDS_TRANSFORM = {
    1: [Atom.SALT],  # Requires calcification from a cardinal
    2: [Atom.SALT, Atom.TIN],  # Salt via calcification, Tin via projection
    3: [Atom.SALT, Atom.TIN, Atom.IRON],
    4: [Atom.SALT, Atom.TIN, Atom.IRON, Atom.COPPER, Atom.VITAE, Atom.MORS],
    5: [Atom.SALT, Atom.TIN, Atom.IRON, Atom.COPPER, Atom.SILVER, Atom.GOLD,
        Atom.VITAE, Atom.MORS, Atom.QUINTESSENCE],
}


def _sample_output_atoms(
    level: int,
    rng: random.Random,
) -> list[Atom]:
    """Sample a set of output atoms appropriate for the complexity level.

    Guarantees at least one atom requires a transformation so that puzzles
    aren't pure pass-throughs.
    """
    pool = LEVEL_ATOMS[level]
    min_atoms, max_atoms = LEVEL_ATOM_COUNTS[level]
    n = rng.randint(min_atoms, max_atoms)

    atoms = [rng.choice(pool) for _ in range(n)]

    # Ensure at least one atom requires a transformation
    transform_pool = _NEEDS_TRANSFORM[level]
    if not any(a in transform_pool for a in atoms):
        # Replace one atom with one that requires a transformation
        atoms[rng.randint(0, len(atoms) - 1)] = rng.choice(transform_pool)

    return atoms


def _partition_atoms(
    atoms: list[Atom],
    num_groups: int,
    rng: random.Random,
) -> list[list[Atom]]:
    """Split atoms into num_groups non-empty groups."""
    if num_groups <= 1:
        return [atoms]

    shuffled = list(atoms)
    rng.shuffle(shuffled)

    # Ensure each group has at least one atom
    groups: list[list[Atom]] = [[] for _ in range(num_groups)]
    for i in range(num_groups):
        if i < len(shuffled):
            groups[i].append(shuffled[i])

    # Distribute remaining atoms randomly
    for atom in shuffled[num_groups:]:
        groups[rng.randint(0, num_groups - 1)].append(atom)

    # Remove empty groups
    return [g for g in groups if g]


def _group_inputs_into_molecules(
    input_atoms: list[Atom],
    rng: random.Random,
) -> list[Molecule]:
    """Group input atoms into molecules. Each distinct atom type becomes
    one or more single-atom input molecules."""
    # Simple approach: each input atom is its own single-atom molecule
    # This matches how most OM puzzles work (inputs are simple reagents)
    molecules: list[Molecule] = []
    counts = Counter(input_atoms)

    for atom_type, count in counts.items():
        # Each unique atom type becomes one input molecule
        # (the puzzle provides count copies per cycle)
        molecules.append(build_single_atom_molecule(atom_type))

    return molecules


def generate_puzzle(
    complexity_level: int = 1,
    seed: int | None = None,
    name: str | None = None,
) -> Puzzle:
    """Generate a random, valid Opus Magnum puzzle.

    Args:
        complexity_level: 1 (simplest) to 5 (hardest).
            1: Single transmutation, 1-2 atoms, one glyph
            2: Multi-step chains, 2-4 atoms
            3: Bonded outputs, 3-6 atoms
            4: Mixed transformations + bonding, 4-8 atoms
            5: Multi-output, triplex bonds, quintessence, 6-12 atoms
        seed: Random seed for reproducibility.
        name: Puzzle name. Auto-generated if None.

    Returns:
        A valid Puzzle object with raw_bytes set.
    """
    if complexity_level not in range(1, 6):
        raise ValueError(f"complexity_level must be 1-5, got {complexity_level}")

    rng = random.Random(seed)

    # Step 1: Sample output atom types
    output_atoms = _sample_output_atoms(complexity_level, rng)

    # Step 2: Determine transformations and derive inputs
    # Force certain atoms to be synthesized (not passed through as direct inputs)
    # so that puzzles actually require glyph usage.
    force_synthesis: frozenset[Atom] | None = None
    if complexity_level <= 2:
        # At low levels, force Salt to be synthesized via calcification
        force_synthesis = frozenset({Atom.SALT})

    try:
        input_atoms, transforms = derive_inputs_for_output(
            output_atoms, force_synthesis=force_synthesis
        )
    except ValueError:
        # If synthesis fails, fall back to simpler output
        output_atoms = [Atom.SALT]
        input_atoms, transforms = derive_inputs_for_output(
            output_atoms, force_synthesis=force_synthesis
        )

    # Step 3: Decide on bonds and molecule structure
    use_bonds = complexity_level >= 3
    use_triplex = complexity_level >= 5
    bond_prob = 0.8 if use_bonds else 0.0
    triplex_prob = 0.3 if use_triplex else 0.0

    # Step 4: Build output molecule(s)
    min_outputs, max_outputs = LEVEL_OUTPUT_COUNTS[complexity_level]
    num_outputs = rng.randint(min_outputs, max_outputs)
    atom_groups = _partition_atoms(output_atoms, num_outputs, rng)

    output_molecules: list[Molecule] = []
    for group in atom_groups:
        mol = build_molecule(group, bond_prob, triplex_prob, rng)
        output_molecules.append(mol)

    # Step 5: Build input molecules
    input_molecules = _group_inputs_into_molecules(input_atoms, rng)

    # Step 6: Compute required tool flags
    tool_flags = BASE_TOOLS

    # Add glyph-specific flags based on transformations
    tool_flags |= required_tools_for_path(transforms)

    # Add bonding flags if output has bonds
    has_normal_bonds = any(
        b.bond_type == BondType.NORMAL
        for mol in output_molecules
        for b in mol.bonds
    )
    has_triplex_bonds = any(
        b.bond_type == BondType.TRIPLEX
        for mol in output_molecules
        for b in mol.bonds
    )
    if has_normal_bonds:
        tool_flags |= ToolFlags.BONDER
    if has_triplex_bonds:
        tool_flags |= ToolFlags.TRIPLEX_BONDER

    # Always include disposal for convenience
    tool_flags |= ToolFlags.DISPOSAL

    # Step 7: Generate name
    if name is None:
        name = f"GENERATED-L{complexity_level}-{seed or rng.randint(0, 99999):05d}"

    # Step 8: Create puzzle
    puzzle = Puzzle(
        name=name,
        creator_id=0,
        allowed_tools=tool_flags,
        inputs=input_molecules,
        outputs=output_molecules,
        output_scale=1,
        is_production=False,
        raw_bytes=b"",
    )

    # Set raw_bytes by serializing
    puzzle.raw_bytes = write_puzzle(puzzle)

    return puzzle


def generate_puzzle_batch(
    count: int,
    complexity_level: int = 1,
    base_seed: int = 0,
) -> list[Puzzle]:
    """Generate a batch of puzzles with sequential seeds."""
    return [
        generate_puzzle(complexity_level=complexity_level, seed=base_seed + i)
        for i in range(count)
    ]
