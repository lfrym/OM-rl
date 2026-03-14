"""Vendored Opus Magnum library for benchmark use."""

# Don't eagerly import verifier or leaderboard (they have external deps).
from .models import *
from .puzzle_parser import parse_puzzle
from .solution_writer import write_solution
from .text_format import puzzle_to_text, parse_text_solution, SOLUTION_FORMAT_SPEC
