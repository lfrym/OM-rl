"""Core data models for Opus Magnum puzzles and solutions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag


# ── Hex grid ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HexVector:
    """Axial hex coordinate (u, v)."""

    u: int
    v: int

    def __add__(self, other: HexVector) -> HexVector:
        return HexVector(self.u + other.u, self.v + other.v)

    def __sub__(self, other: HexVector) -> HexVector:
        return HexVector(self.u - other.u, self.v - other.v)

    def __neg__(self) -> HexVector:
        return HexVector(-self.u, -self.v)

    def rotate_cw(self) -> HexVector:
        """Rotate 60° clockwise."""
        return HexVector(-self.v, self.u + self.v)

    def rotate_ccw(self) -> HexVector:
        """Rotate 60° counter-clockwise."""
        return HexVector(self.u + self.v, -self.u)

    def rotate(self, steps: int) -> HexVector:
        """Rotate by steps * 60°. Positive = CW."""
        v = self
        steps = steps % 6
        for _ in range(steps):
            v = v.rotate_cw()
        return v

    def hex_distance(self, other: HexVector) -> int:
        du = self.u - other.u
        dv = self.v - other.v
        return max(abs(du), abs(dv), abs(du + dv))


# Six unit directions on the hex grid (starting East, going CW).
HEX_DIRECTIONS = [
    HexVector(1, 0),  # 0: E
    HexVector(0, 1),  # 1: SE
    HexVector(-1, 1),  # 2: SW
    HexVector(-1, 0),  # 3: W
    HexVector(0, -1),  # 4: NW
    HexVector(1, -1),  # 5: NE
]


# ── Atoms / Elements ────────────────────────────────────────────────────────


class Atom(IntEnum):
    """Element types, matching the binary puzzle format byte values."""

    SALT = 1
    AIR = 2
    EARTH = 3
    FIRE = 4
    WATER = 5
    QUICKSILVER = 6
    GOLD = 7
    SILVER = 8
    COPPER = 9
    IRON = 10
    TIN = 11
    LEAD = 12
    VITAE = 13
    MORS = 14
    REPEAT = 15
    QUINTESSENCE = 16

    @property
    def is_cardinal(self) -> bool:
        return self in (Atom.AIR, Atom.FIRE, Atom.WATER, Atom.EARTH)

    @property
    def is_metal(self) -> bool:
        return self in (Atom.LEAD, Atom.TIN, Atom.IRON, Atom.COPPER, Atom.SILVER, Atom.GOLD)

    @property
    def symbol(self) -> str:
        return _ATOM_SYMBOLS[self]


_ATOM_SYMBOLS = {
    Atom.SALT: "Sa",
    Atom.AIR: "Ai",
    Atom.EARTH: "Ea",
    Atom.FIRE: "Fi",
    Atom.WATER: "Wa",
    Atom.QUICKSILVER: "Qs",
    Atom.GOLD: "Au",
    Atom.SILVER: "Ag",
    Atom.COPPER: "Cu",
    Atom.IRON: "Fe",
    Atom.TIN: "Sn",
    Atom.LEAD: "Pb",
    Atom.VITAE: "Vi",
    Atom.MORS: "Mo",
    Atom.REPEAT: "..",
    Atom.QUINTESSENCE: "Qt",
}

# Metal purification order (projection / purification chain).
METAL_ORDER = [Atom.LEAD, Atom.TIN, Atom.IRON, Atom.COPPER, Atom.SILVER, Atom.GOLD]


# ── Bonds ───────────────────────────────────────────────────────────────────


class BondType(IntFlag):
    NORMAL = 1
    TRIPLEX_RED = 2
    TRIPLEX_BLACK = 4
    TRIPLEX_YELLOW = 8
    TRIPLEX = TRIPLEX_RED | TRIPLEX_BLACK | TRIPLEX_YELLOW


@dataclass(frozen=True, slots=True)
class Bond:
    """A bond between two atom positions in a molecule."""

    from_pos: HexVector
    to_pos: HexVector
    bond_type: BondType


# ── Molecules (used in puzzle I/O definitions) ──────────────────────────────


@dataclass(slots=True)
class Molecule:
    """A molecule as defined in a puzzle file (reagent or product)."""

    atoms: list[tuple[HexVector, Atom]]
    bonds: list[Bond]


# ── Puzzles ─────────────────────────────────────────────────────────────────


class ToolFlags(IntFlag):
    """Bitfield for which tools/glyphs are available in a puzzle."""

    ARM = 1 << 0
    MULTI_ARM = 1 << 1
    PISTON = 1 << 2
    TRACK = 1 << 3
    BONDER = 1 << 8
    UNBONDER = 1 << 9
    MULTI_BONDER = 1 << 10
    TRIPLEX_BONDER = 1 << 11
    CALCIFICATION = 1 << 12
    DUPLICATION = 1 << 13
    PROJECTION = 1 << 14
    PURIFICATION = 1 << 15
    ANIMISMUS = 1 << 16
    DISPOSAL = 1 << 17
    QUINTESSENCE_GLYPHS = 1 << 18  # unification + dispersion
    GRAB_AND_TURN = 1 << 22
    DROP = 1 << 23
    RESET = 1 << 24
    REPEAT = 1 << 25
    PIVOT = 1 << 26
    VAN_BERLOS_WHEEL = 1 << 28


@dataclass(slots=True)
class Puzzle:
    """A parsed Opus Magnum puzzle."""

    name: str
    creator_id: int
    allowed_tools: ToolFlags
    inputs: list[Molecule]
    outputs: list[Molecule]
    output_scale: int
    is_production: bool

    # We store the raw bytes so we can pass them to omsim.
    raw_bytes: bytes = field(default=b"", repr=False)


# ── Solution components ─────────────────────────────────────────────────────


class ArmType(str, Enum):
    ARM1 = "arm1"
    ARM2 = "arm2"
    ARM3 = "arm3"
    ARM6 = "arm6"
    PISTON = "piston"
    VAN_BERLO = "baron"


ARM_COSTS = {
    ArmType.ARM1: 20,
    ArmType.ARM2: 30,
    ArmType.ARM3: 30,
    ArmType.ARM6: 30,
    ArmType.PISTON: 40,
    ArmType.VAN_BERLO: 30,
}


class GlyphType(str, Enum):
    BONDER = "bonder"
    MULTI_BONDER = "bonder-speed"
    TRIPLEX_BONDER = "bonder-prisma"
    UNBONDER = "unbonder"
    CALCIFICATION = "glyph-calcification"
    DUPLICATION = "glyph-duplication"
    PROJECTION = "glyph-projection"
    PURIFICATION = "glyph-purification"
    ANIMISMUS = "glyph-life-and-death"
    DISPOSAL = "glyph-disposal"
    EQUILIBRIUM = "glyph-marker"
    UNIFICATION = "glyph-unification"
    DISPERSION = "glyph-dispersion"


GLYPH_COSTS = {
    GlyphType.CALCIFICATION: 10,
    GlyphType.BONDER: 10,
    GlyphType.UNBONDER: 10,
    GlyphType.TRIPLEX_BONDER: 20,
    GlyphType.ANIMISMUS: 20,
    GlyphType.PROJECTION: 20,
    GlyphType.DISPERSION: 20,
    GlyphType.PURIFICATION: 20,
    GlyphType.DUPLICATION: 20,
    GlyphType.UNIFICATION: 20,
    GlyphType.MULTI_BONDER: 30,
    GlyphType.DISPOSAL: 0,
    GlyphType.EQUILIBRIUM: 0,
}

# Footprints: list of hex offsets relative to the glyph origin.
# The origin (0,0) is always the LAST entry (matches omsim convention).
GLYPH_FOOTPRINTS: dict[GlyphType, list[HexVector]] = {
    GlyphType.CALCIFICATION: [HexVector(0, 0)],
    GlyphType.ANIMISMUS: [HexVector(0, 1), HexVector(1, 0), HexVector(1, -1), HexVector(0, 0)],
    GlyphType.PROJECTION: [HexVector(1, 0), HexVector(0, 0)],
    GlyphType.DISPERSION: [
        HexVector(1, 0),
        HexVector(1, -1),
        HexVector(0, -1),
        HexVector(-1, 0),
        HexVector(0, 0),
    ],
    GlyphType.PURIFICATION: [HexVector(1, 0), HexVector(0, 1), HexVector(0, 0)],
    GlyphType.DUPLICATION: [HexVector(1, 0), HexVector(0, 0)],
    GlyphType.UNIFICATION: [
        HexVector(0, 1),
        HexVector(-1, 1),
        HexVector(0, -1),
        HexVector(1, -1),
        HexVector(0, 0),
    ],
    GlyphType.BONDER: [HexVector(1, 0), HexVector(0, 0)],
    GlyphType.UNBONDER: [HexVector(1, 0), HexVector(0, 0)],
    GlyphType.TRIPLEX_BONDER: [HexVector(1, 0), HexVector(0, 1), HexVector(0, 0)],
    GlyphType.MULTI_BONDER: [HexVector(1, 0), HexVector(0, -1), HexVector(-1, 1), HexVector(0, 0)],
    GlyphType.DISPOSAL: [
        HexVector(1, 0),
        HexVector(0, 1),
        HexVector(-1, 1),
        HexVector(-1, 0),
        HexVector(0, -1),
        HexVector(1, -1),
        HexVector(0, 0),
    ],
    GlyphType.EQUILIBRIUM: [HexVector(0, 0)],
}


class Instruction(str, Enum):
    """Arm instruction codes (matching the solution file encoding)."""

    ROTATE_CW = "R"
    ROTATE_CCW = "r"
    EXTEND = "E"
    RETRACT = "e"
    GRAB = "G"
    DROP = "g"
    PIVOT_CW = "P"
    PIVOT_CCW = "p"
    FORWARD = "A"
    BACK = "a"
    REPEAT = "C"
    RESET = "X"
    NOOP = "O"


TRACK_COST_PER_HEX = 5


@dataclass(slots=True)
class Arm:
    arm_type: ArmType
    position: HexVector
    rotation: int  # 0-5 (multiples of 60°)
    extension: int  # arm length (1 for arm1, etc.)
    instructions: dict[int, Instruction]  # cycle -> instruction
    arm_id: int = 0


@dataclass(slots=True)
class Glyph:
    glyph_type: GlyphType
    position: HexVector
    rotation: int  # 0-5


@dataclass(slots=True)
class Track:
    positions: list[HexVector]


class IOType(str, Enum):
    INPUT = "input"
    OUTPUT = "out-std"
    INFINITE = "out-rep"


@dataclass(slots=True)
class IOPart:
    io_type: IOType
    position: HexVector
    rotation: int
    molecule_index: int


@dataclass(slots=True)
class Solution:
    """A complete Opus Magnum solution."""

    puzzle_name: str
    solution_name: str
    arms: list[Arm]
    glyphs: list[Glyph]
    tracks: list[Track]
    ios: list[IOPart]  # input/output placements

    def estimated_cost(self) -> int:
        cost = 0
        for arm in self.arms:
            cost += ARM_COSTS[arm.arm_type]
        for glyph in self.glyphs:
            cost += GLYPH_COSTS[glyph.glyph_type]
        for track in self.tracks:
            cost += len(track.positions) * TRACK_COST_PER_HEX
        return cost
