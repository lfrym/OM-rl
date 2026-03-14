"""Step-by-step replay debugger for Opus Magnum solutions.

Uses omsim's trace() to get per-cycle simulation states, then provides
navigation tools to step through the replay and inspect board state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vendor.opus_magnum.models import Puzzle
from vendor.opus_magnum.verifier import trace, VerificationError


@dataclass
class CycleState:
    """State of the simulation at a single cycle."""

    cycle: int
    collision: bool
    atoms: list[dict[str, Any]]  # [{u, v, type, grabbed}, ...]
    arms: list[dict[str, Any]]  # [{u, v, rotation, grabbing}, ...]
    raw: dict[str, Any]


class ReplayDebugger:
    """Navigate through a solution's execution cycle by cycle.

    Usage:
        debugger = ReplayDebugger(puzzle_bytes, solution_bytes)
        debugger.load(max_cycles=200)
        print(debugger.current())   # Cycle 0
        debugger.forward()           # Cycle 1
        debugger.forward(5)          # Cycle 6
        debugger.back()              # Cycle 5
        debugger.goto(10)            # Cycle 10
    """

    def __init__(self, puzzle_bytes: bytes, solution_bytes: bytes):
        self._puzzle_bytes = puzzle_bytes
        self._solution_bytes = solution_bytes
        self._states: list[CycleState] = []
        self._pos: int = 0

    def load(self, max_cycles: int = 200) -> int:
        """Load the trace. Returns number of cycles traced."""
        raw_trace = trace(
            self._puzzle_bytes,
            self._solution_bytes,
            max_cycles=max_cycles,
        )
        self._states = []
        for entry in raw_trace:
            self._states.append(CycleState(
                cycle=entry.get("cycle", len(self._states)),
                collision=entry.get("collision", False),
                atoms=entry.get("atoms", []),
                arms=entry.get("arms", []),
                raw=entry,
            ))
        self._pos = 0
        return len(self._states)

    @property
    def num_cycles(self) -> int:
        return len(self._states)

    @property
    def position(self) -> int:
        return self._pos

    def current(self) -> CycleState | None:
        """Get the current cycle state."""
        if not self._states:
            return None
        return self._states[self._pos]

    def forward(self, steps: int = 1) -> CycleState | None:
        """Advance forward by N cycles."""
        self._pos = min(self._pos + steps, len(self._states) - 1)
        return self.current()

    def back(self, steps: int = 1) -> CycleState | None:
        """Go back by N cycles."""
        self._pos = max(self._pos - steps, 0)
        return self.current()

    def goto(self, cycle: int) -> CycleState | None:
        """Jump to a specific cycle number."""
        self._pos = max(0, min(cycle, len(self._states) - 1))
        return self.current()

    def format_state(self, state: CycleState | None = None) -> str:
        """Format a cycle state as readable text."""
        if state is None:
            state = self.current()
        if state is None:
            return "(no trace loaded)"

        lines = [f"=== Cycle {state.cycle} ==="]
        if state.collision:
            lines.append("!! COLLISION DETECTED !!")

        lines.append(f"Atoms ({len(state.atoms)}):")
        for atom in state.atoms:
            grabbed = " [GRABBED]" if atom.get("grabbed") else ""
            lines.append(f"  ({atom.get('u', '?')}, {atom.get('v', '?')}) "
                         f"type={atom.get('type', '?')}{grabbed}")

        lines.append(f"Arms ({len(state.arms)}):")
        for arm in state.arms:
            grabbing = " [GRABBING]" if arm.get("grabbing") else ""
            lines.append(f"  ({arm.get('u', '?')}, {arm.get('v', '?')}) "
                         f"rot={arm.get('rotation', '?')}{grabbing}")

        return "\n".join(lines)

    def find_error_cycle(self) -> int | None:
        """Find the first cycle where a collision occurs."""
        for state in self._states:
            if state.collision:
                return state.cycle
        return None

    def summary(self) -> str:
        """Get a summary of the trace."""
        if not self._states:
            return "No trace loaded."

        error_cycle = self.find_error_cycle()
        lines = [
            f"Trace: {len(self._states)} cycles",
            f"Current position: cycle {self._pos}",
        ]
        if error_cycle is not None:
            lines.append(f"First error at cycle {error_cycle}")
        else:
            lines.append("No errors detected in trace")
        return "\n".join(lines)
