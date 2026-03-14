#!/usr/bin/env python3
"""Interactive solution debugger — step through a solution cycle by cycle."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from vendor.opus_magnum.puzzle_parser import parse_puzzle
from vendor.opus_magnum.text_format import parse_text_solution
from vendor.opus_magnum.solution_writer import write_solution
from vendor.opus_magnum.verifier import VerificationError
from om_rl.env.interactive import ReplayDebugger


def main():
    parser = argparse.ArgumentParser(description="Step through an Opus Magnum solution")
    parser.add_argument("puzzle", help="Path to .puzzle file")
    parser.add_argument("solution", help="Path to solution text file")
    parser.add_argument("--max-cycles", type=int, default=200, help="Max cycles to trace")
    args = parser.parse_args()

    # Load puzzle and solution
    puzzle = parse_puzzle(args.puzzle)
    solution_text = Path(args.solution).read_text()
    solution = parse_text_solution(solution_text, puzzle.name)
    solution_bytes = write_solution(solution)

    print(f"Puzzle: {puzzle.name}")
    print(f"Loading trace (max {args.max_cycles} cycles)...")

    try:
        debugger = ReplayDebugger(puzzle.raw_bytes, solution_bytes)
        num = debugger.load(max_cycles=args.max_cycles)
        print(f"Loaded {num} cycles")
        print(debugger.summary())
    except VerificationError as e:
        print(f"Verification error: {e}")
        print("(Trace may be partial)")
        return

    # Interactive loop
    print("\nCommands: f=forward, b=back, g <N>=goto cycle N, s=show state, q=quit")
    while True:
        try:
            cmd = input(f"\n[cycle {debugger.position}] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "f":
            debugger.forward()
            print(debugger.format_state())
        elif cmd == "b":
            debugger.back()
            print(debugger.format_state())
        elif cmd.startswith("g "):
            try:
                n = int(cmd[2:])
                debugger.goto(n)
                print(debugger.format_state())
            except ValueError:
                print("Usage: g <cycle_number>")
        elif cmd == "s":
            print(debugger.format_state())
        elif cmd == "":
            debugger.forward()
            print(debugger.format_state())
        else:
            print("Unknown command. f=forward, b=back, g <N>=goto, s=show, q=quit")


if __name__ == "__main__":
    main()
