"""Tests for puzzle generation pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from om_rl.puzzle_gen.chemistry import (
    ALL_TRANSFORMATIONS,
    CARDINALS,
    METAL_CHAIN,
    derive_inputs_for_output,
    find_synthesis_path,
    transformation_depth,
)
from om_rl.puzzle_gen.generator import generate_puzzle, generate_puzzle_batch
from om_rl.puzzle_gen.molecule_builder import (
    build_molecule,
    build_linear_molecule,
    build_single_atom_molecule,
)
from om_rl.puzzle_gen.puzzle_writer import write_puzzle
from om_rl.puzzle_gen.validator import validate_puzzle
from vendor.opus_magnum.models import Atom, BondType, HexVector
from vendor.opus_magnum.puzzle_parser import parse_puzzle


class TestChemistry:
    def test_transformations_exist(self):
        assert len(ALL_TRANSFORMATIONS) > 0

    def test_cardinals_to_salt(self):
        for elem in CARDINALS:
            path = find_synthesis_path(Atom.SALT, frozenset({elem}))
            assert path is not None, f"Cannot synthesize salt from {elem}"

    def test_metal_chain(self):
        for i in range(1, len(METAL_CHAIN)):
            target = METAL_CHAIN[i]
            path = find_synthesis_path(
                target, frozenset({Atom.QUICKSILVER, Atom.LEAD})
            )
            assert path is not None, f"Cannot synthesize {target} from QS+Lead"

    def test_quintessence(self):
        path = find_synthesis_path(
            Atom.QUINTESSENCE,
            frozenset({Atom.AIR, Atom.FIRE, Atom.WATER, Atom.EARTH}),
        )
        assert path is not None

    def test_vitae_mors(self):
        path = find_synthesis_path(Atom.VITAE, frozenset({Atom.SALT}))
        assert path is not None

    def test_derive_inputs(self):
        inputs, transforms = derive_inputs_for_output([Atom.GOLD])
        assert len(inputs) > 0
        assert len(transforms) > 0

    def test_transformation_depth(self):
        depth = transformation_depth(
            [Atom.SALT], frozenset({Atom.WATER})
        )
        assert depth == 1  # One calcification step

        depth = transformation_depth(
            [Atom.WATER], frozenset({Atom.WATER})
        )
        assert depth == 0  # Already available


class TestMoleculeBuilder:
    def test_single_atom(self):
        mol = build_single_atom_molecule(Atom.SALT)
        assert len(mol.atoms) == 1
        assert mol.atoms[0] == (HexVector(0, 0), Atom.SALT)

    def test_multi_atom(self):
        mol = build_molecule([Atom.SALT, Atom.WATER, Atom.FIRE])
        assert len(mol.atoms) == 3

    def test_bonded(self):
        mol = build_molecule(
            [Atom.SALT, Atom.WATER],
            bond_probability=1.0,
        )
        assert len(mol.atoms) == 2
        assert len(mol.bonds) == 1  # Two adjacent atoms, should bond

    def test_linear(self):
        mol = build_linear_molecule([Atom.SALT, Atom.WATER, Atom.FIRE])
        assert len(mol.atoms) == 3
        assert len(mol.bonds) == 2


class TestPuzzleWriter:
    def test_roundtrip_generated(self):
        puzzle = generate_puzzle(complexity_level=1, seed=42)
        written = write_puzzle(puzzle)
        parsed = parse_puzzle(written)
        assert parsed.name == puzzle.name
        assert len(parsed.inputs) == len(puzzle.inputs)
        assert len(parsed.outputs) == len(puzzle.outputs)

    def test_roundtrip_campaign(self):
        import glob

        files = glob.glob("puzzles/campaign/*.puzzle")
        for f in files[:5]:
            p = parse_puzzle(f)
            written = write_puzzle(p)
            p2 = parse_puzzle(written)
            assert p2.name == p.name


class TestGenerator:
    def test_level_1(self):
        puzzle = generate_puzzle(complexity_level=1, seed=0)
        assert validate_puzzle(puzzle)
        assert len(puzzle.outputs) == 1
        total_atoms = sum(len(m.atoms) for m in puzzle.outputs)
        assert 1 <= total_atoms <= 2

    def test_level_3_has_bonds(self):
        # Level 3 should sometimes produce bonds
        has_bonds = False
        for seed in range(20):
            p = generate_puzzle(complexity_level=3, seed=seed)
            if any(mol.bonds for mol in p.outputs):
                has_bonds = True
                break
        assert has_bonds, "Level 3 should produce bonded outputs"

    def test_batch(self):
        puzzles = generate_puzzle_batch(10, complexity_level=2, base_seed=0)
        assert len(puzzles) == 10
        for p in puzzles:
            assert validate_puzzle(p)

    def test_all_levels(self):
        for level in range(1, 6):
            puzzles = generate_puzzle_batch(5, complexity_level=level, base_seed=level * 100)
            for p in puzzles:
                result = validate_puzzle(p)
                assert result, f"Level {level} puzzle invalid: {result.issues}"

    def test_omsim_accepts_generated(self):
        """Verify omsim can load generated puzzles (doesn't crash)."""
        from vendor.opus_magnum.verifier import Verifier, VerificationError
        from vendor.opus_magnum.solution_writer import write_solution
        from vendor.opus_magnum.models import (
            Solution, Arm, ArmType, IOPart, IOType, Instruction,
        )

        for level in range(1, 4):
            puzzle = generate_puzzle(complexity_level=level, seed=level)
            # Minimal dummy solution
            sol = Solution(
                puzzle_name=puzzle.name,
                solution_name="test",
                arms=[Arm(ArmType.ARM1, HexVector(3, 0), 0, 1, {}, 0)],
                glyphs=[],
                tracks=[],
                ios=[
                    IOPart(IOType.INPUT, HexVector(-3, 0), 0, 0),
                    IOPart(IOType.OUTPUT, HexVector(6, 0), 0, 0),
                ],
            )
            sol_bytes = write_solution(sol)
            try:
                with Verifier(puzzle.raw_bytes, sol_bytes, cycle_limit=10) as v:
                    v.evaluate()
            except VerificationError:
                pass  # Expected — dummy solution won't solve the puzzle


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
