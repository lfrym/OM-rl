"""Opus Magnum atom transformation graph.

Models all glyph-based transformations as a directed graph so we can:
1. Determine which glyphs are needed to convert input atoms into output atoms
2. Work backward from desired outputs to find valid inputs
3. Check chemical reachability (solvability)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

from vendor.opus_magnum.models import Atom, GlyphType, ToolFlags

# Metal chain order (projection / purification)
METAL_CHAIN = [Atom.LEAD, Atom.TIN, Atom.IRON, Atom.COPPER, Atom.SILVER, Atom.GOLD]
METAL_RANK = {m: i for i, m in enumerate(METAL_CHAIN)}

CARDINALS = frozenset({Atom.AIR, Atom.EARTH, Atom.FIRE, Atom.WATER})


@dataclass(frozen=True)
class Transformation:
    """A single glyph-based atom transformation.

    consumed: atoms consumed (multiset as tuple of (Atom, count))
    produced: atoms produced (multiset as tuple of (Atom, count))
    glyph: which glyph performs this
    """

    consumed: tuple[tuple[Atom, int], ...]
    produced: tuple[tuple[Atom, int], ...]
    glyph: GlyphType
    name: str = ""


def build_transformations() -> list[Transformation]:
    """Return all possible single-step atom transformations in Opus Magnum."""
    transforms: list[Transformation] = []

    # Calcification: any cardinal element -> Salt
    for elem in CARDINALS:
        transforms.append(Transformation(
            consumed=((elem, 1),),
            produced=((Atom.SALT, 1),),
            glyph=GlyphType.CALCIFICATION,
            name=f"calcify_{elem.name.lower()}",
        ))

    # Projection: Quicksilver + Metal_n -> Metal_{n+1}
    # (consumes quicksilver, upgrades the metal one step)
    for i in range(len(METAL_CHAIN) - 1):
        transforms.append(Transformation(
            consumed=((Atom.QUICKSILVER, 1), (METAL_CHAIN[i], 1)),
            produced=((METAL_CHAIN[i + 1], 1),),
            glyph=GlyphType.PROJECTION,
            name=f"project_{METAL_CHAIN[i].name.lower()}_to_{METAL_CHAIN[i+1].name.lower()}",
        ))

    # Purification: 2x Metal_n -> Metal_{n+1}
    for i in range(len(METAL_CHAIN) - 1):
        transforms.append(Transformation(
            consumed=((METAL_CHAIN[i], 2),),
            produced=((METAL_CHAIN[i + 1], 1),),
            glyph=GlyphType.PURIFICATION,
            name=f"purify_{METAL_CHAIN[i].name.lower()}_to_{METAL_CHAIN[i+1].name.lower()}",
        ))

    # Animismus (Life and Death): 2 Salt -> Vitae + Mors
    transforms.append(Transformation(
        consumed=((Atom.SALT, 2),),
        produced=((Atom.VITAE, 1), (Atom.MORS, 1)),
        glyph=GlyphType.ANIMISMUS,
        name="animismus",
    ))

    # Unification: Air + Fire + Water + Earth -> Quintessence
    transforms.append(Transformation(
        consumed=((Atom.AIR, 1), (Atom.FIRE, 1), (Atom.WATER, 1), (Atom.EARTH, 1)),
        produced=((Atom.QUINTESSENCE, 1),),
        glyph=GlyphType.UNIFICATION,
        name="unify",
    ))

    # Dispersion: Quintessence -> Air + Fire + Water + Earth
    transforms.append(Transformation(
        consumed=((Atom.QUINTESSENCE, 1),),
        produced=((Atom.AIR, 1), (Atom.FIRE, 1), (Atom.WATER, 1), (Atom.EARTH, 1)),
        glyph=GlyphType.DISPERSION,
        name="disperse",
    ))

    # Duplication: Salt + cardinal_on_adjacent_tile -> Salt + 2x cardinal
    # (Salt is consumed and reproduced; net effect is duplicating the cardinal)
    for elem in CARDINALS:
        transforms.append(Transformation(
            consumed=((Atom.SALT, 1), (elem, 1)),
            produced=((elem, 2),),
            glyph=GlyphType.DUPLICATION,
            name=f"duplicate_{elem.name.lower()}",
        ))

    return transforms


# Singleton list of all transformations
ALL_TRANSFORMATIONS = build_transformations()

# Lookup: which transformations produce a given atom type?
PRODUCES: dict[Atom, list[Transformation]] = defaultdict(list)
for _t in ALL_TRANSFORMATIONS:
    for _atom, _count in _t.produced:
        PRODUCES[_atom].append(_t)

# Lookup: glyph -> ToolFlag mapping
GLYPH_TO_TOOL: dict[GlyphType, ToolFlags] = {
    GlyphType.CALCIFICATION: ToolFlags.CALCIFICATION,
    GlyphType.PROJECTION: ToolFlags.PROJECTION,
    GlyphType.PURIFICATION: ToolFlags.PURIFICATION,
    GlyphType.ANIMISMUS: ToolFlags.ANIMISMUS,
    GlyphType.UNIFICATION: ToolFlags.QUINTESSENCE_GLYPHS,
    GlyphType.DISPERSION: ToolFlags.QUINTESSENCE_GLYPHS,
    GlyphType.DUPLICATION: ToolFlags.DUPLICATION,
    GlyphType.BONDER: ToolFlags.BONDER,
    GlyphType.UNBONDER: ToolFlags.UNBONDER,
    GlyphType.MULTI_BONDER: ToolFlags.MULTI_BONDER,
    GlyphType.TRIPLEX_BONDER: ToolFlags.TRIPLEX_BONDER,
    GlyphType.DISPOSAL: ToolFlags.DISPOSAL,
}


def find_synthesis_path(
    target_atom: Atom,
    available_inputs: frozenset[Atom],
    max_depth: int = 10,
) -> list[Transformation] | None:
    """Find a sequence of transformations to produce target_atom from available_inputs.

    Returns the transformation chain, or None if unreachable.
    Uses BFS to find shortest path.
    """
    if target_atom in available_inputs:
        return []

    from collections import deque

    # BFS: state is the atom we're trying to produce
    # We search backward: what transformations produce this atom?
    queue: deque[tuple[Atom, list[Transformation]]] = deque()
    queue.append((target_atom, []))
    visited: set[Atom] = {target_atom}

    while queue:
        current, path = queue.popleft()
        if len(path) >= max_depth:
            continue

        for transform in PRODUCES.get(current, []):
            new_path = [transform] + path
            # Check if all consumed atoms are available as direct inputs
            all_available = True
            for atom, count in transform.consumed:
                if atom not in available_inputs:
                    all_available = False
                    if atom not in visited:
                        visited.add(atom)
                        queue.append((atom, new_path))

            if all_available:
                return new_path

    return None


def required_glyphs_for_path(path: list[Transformation]) -> set[GlyphType]:
    """Get the set of glyphs needed for a transformation path."""
    return {t.glyph for t in path}


def required_tools_for_path(path: list[Transformation]) -> ToolFlags:
    """Get the ToolFlags needed for a transformation path."""
    flags = ToolFlags(0)
    for t in path:
        if t.glyph in GLYPH_TO_TOOL:
            flags |= GLYPH_TO_TOOL[t.glyph]
    return flags


def derive_inputs_for_output(
    output_atoms: list[Atom],
    allowed_glyphs: set[GlyphType] | None = None,
    force_synthesis: frozenset[Atom] | None = None,
) -> tuple[list[Atom], list[Transformation]]:
    """Given desired output atoms, work backward to find what input atoms are needed.

    Args:
        output_atoms: Desired output atom types.
        allowed_glyphs: If set, only these glyphs may be used.
        force_synthesis: Atom types that must be synthesized even if they
            could be provided directly as inputs. E.g., frozenset({Atom.SALT})
            forces Salt to be produced via calcification from a cardinal.

    Returns (input_atoms, transformations_needed).
    """
    needed_inputs: list[Atom] = []
    all_transforms: list[Transformation] = []

    # Base set of atoms that can be puzzle inputs without transformation
    base_direct = frozenset({
        Atom.SALT, Atom.AIR, Atom.EARTH, Atom.FIRE, Atom.WATER,
        Atom.QUICKSILVER, Atom.LEAD,
    })

    for target in output_atoms:
        DIRECT_INPUTS = base_direct - (force_synthesis or frozenset())

        if target in DIRECT_INPUTS:
            needed_inputs.append(target)
            continue

        # Find a synthesis path from direct inputs
        path = find_synthesis_path(target, DIRECT_INPUTS)
        if path is None:
            raise ValueError(f"Cannot synthesize {target.name} from basic inputs")

        if allowed_glyphs is not None:
            # Check if path uses only allowed glyphs
            for t in path:
                if t.glyph not in allowed_glyphs:
                    raise ValueError(
                        f"Synthesis of {target.name} requires {t.glyph.name} "
                        f"which is not in allowed glyphs"
                    )

        all_transforms.extend(path)

        # Collect net inputs: atoms consumed by the path that aren't
        # produced by an earlier step in the path.
        # Track what the path produces as intermediate results.
        produced_by_path: dict[Atom, int] = {}
        for t in path:
            # First, consume: deduct from produced_by_path or add to needed_inputs
            for atom, count in t.consumed:
                remaining = count
                if atom in produced_by_path:
                    use = min(produced_by_path[atom], remaining)
                    produced_by_path[atom] -= use
                    remaining -= use
                    if produced_by_path[atom] == 0:
                        del produced_by_path[atom]
                if remaining > 0:
                    needed_inputs.extend([atom] * remaining)
            # Then, produce
            for atom, count in t.produced:
                produced_by_path[atom] = produced_by_path.get(atom, 0) + count

    return needed_inputs, all_transforms


def transformation_depth(
    output_atoms: list[Atom],
    input_atoms: frozenset[Atom],
) -> int:
    """Compute the minimum total number of glyph activations to produce output_atoms
    from input_atoms. Returns 0 if all outputs are directly available as inputs."""
    total = 0
    for target in output_atoms:
        path = find_synthesis_path(target, input_atoms)
        if path is None:
            return -1  # unreachable
        total += len(path)
    return total
