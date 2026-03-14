#!/usr/bin/env python3
"""Generate random Opus Magnum puzzles."""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from om_rl.puzzle_gen.generator import generate_puzzle
from om_rl.puzzle_gen.validator import validate_puzzle
from om_rl.puzzle_gen.puzzle_writer import save_puzzle
from vendor.opus_magnum.text_format import puzzle_to_text


def main():
    parser = argparse.ArgumentParser(description="Generate Opus Magnum puzzles")
    parser.add_argument("--count", type=int, default=10, help="Number of puzzles to generate")
    parser.add_argument("--level", type=int, default=1, help="Complexity level (1-5)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--output-dir", default=None, help="Save puzzles to directory")
    parser.add_argument("--verbose", action="store_true", help="Print full puzzle text")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0

    for i in range(args.count):
        seed = args.seed + i
        try:
            puzzle = generate_puzzle(complexity_level=args.level, seed=seed)
            result = validate_puzzle(puzzle)

            if not result:
                print(f"[INVALID] seed={seed}: {result.issues}")
                failed += 1
                continue

            generated += 1
            in_atoms = [a.symbol for mol in puzzle.inputs for _, a in mol.atoms]
            out_atoms = [a.symbol for mol in puzzle.outputs for _, a in mol.atoms]
            out_bonds = sum(len(mol.bonds) for mol in puzzle.outputs)
            print(f"[OK] {puzzle.name}: {in_atoms} -> {out_atoms} ({out_bonds} bonds)")

            if args.verbose:
                print(puzzle_to_text(puzzle))
                print()

            if output_dir:
                save_puzzle(puzzle, output_dir / f"{puzzle.name}.puzzle")

        except Exception as e:
            print(f"[ERROR] seed={seed}: {e}")
            failed += 1

    print(f"\nGenerated {generated} puzzles ({failed} failed)")


if __name__ == "__main__":
    main()
