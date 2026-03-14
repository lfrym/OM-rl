"""Python wrapper around omsim's libverify shared library."""

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from pathlib import Path


def _find_library() -> Path:
    # Look in project root's lib/ directory (two levels up from vendor/opus_magnum/)
    lib_dir = Path(__file__).resolve().parent.parent.parent / "lib"
    system = platform.system()
    if system == "Darwin":
        path = lib_dir / "libverify.dylib"
    elif system == "Linux":
        path = lib_dir / "libverify.so"
    elif system == "Windows":
        path = lib_dir / "libverify.dll"
    else:
        raise OSError(f"Unsupported platform: {system}")
    if not path.exists():
        # Fallback for Docker sandbox
        for fallback in [Path("/opt/omsim/libverify.so"), Path("/opt/omsim/libverify.dylib")]:
            if fallback.exists():
                return fallback
        raise FileNotFoundError(f"libverify not found at {path}. Run build_omsim.sh first.")
    return path


_lib: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(_find_library()))
        _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    lib.verifier_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.verifier_create.restype = ctypes.c_void_p

    lib.verifier_create_from_bytes.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.verifier_create_from_bytes.restype = ctypes.c_void_p

    lib.verifier_destroy.argtypes = [ctypes.c_void_p]
    lib.verifier_destroy.restype = None

    lib.verifier_set_cycle_limit.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.verifier_set_cycle_limit.restype = None

    lib.verifier_disable_limits.argtypes = [ctypes.c_void_p]
    lib.verifier_disable_limits.restype = None

    lib.verifier_error.argtypes = [ctypes.c_void_p]
    lib.verifier_error.restype = ctypes.c_char_p

    lib.verifier_error_clear.argtypes = [ctypes.c_void_p]
    lib.verifier_error_clear.restype = None

    lib.verifier_error_cycle.argtypes = [ctypes.c_void_p]
    lib.verifier_error_cycle.restype = ctypes.c_int

    lib.verifier_error_location_u.argtypes = [ctypes.c_void_p]
    lib.verifier_error_location_u.restype = ctypes.c_int

    lib.verifier_error_location_v.argtypes = [ctypes.c_void_p]
    lib.verifier_error_location_v.restype = ctypes.c_int

    lib.verifier_evaluate_metric.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.verifier_evaluate_metric.restype = ctypes.c_int

    lib.verifier_evaluate_approximate_metric.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.verifier_evaluate_approximate_metric.restype = ctypes.c_double


@dataclass(frozen=True, slots=True)
class Metrics:
    """Scored metrics for a verified solution."""

    cost: int
    cycles: int
    area: int
    instructions: int


class VerificationError(Exception):
    """Raised when omsim reports an error during verification."""

    def __init__(self, message: str, cycle: int = -1, location: tuple[int, int] | None = None):
        self.cycle = cycle
        self.location = location
        super().__init__(message)


class Verifier:
    """Verify an Opus Magnum solution using omsim's libverify."""

    def __init__(
        self,
        puzzle_bytes: bytes,
        solution_bytes: bytes,
        cycle_limit: int = 100_000,
    ):
        lib = _get_lib()
        self._lib = lib
        self._handle = lib.verifier_create_from_bytes(
            puzzle_bytes,
            len(puzzle_bytes),
            solution_bytes,
            len(solution_bytes),
        )
        if not self._handle:
            raise VerificationError("Failed to create verifier (null handle)")
        self._check_error()
        if cycle_limit > 0:
            lib.verifier_set_cycle_limit(self._handle, cycle_limit)

    def _check_error(self) -> None:
        err = self._lib.verifier_error(self._handle)
        if err:
            msg = err.decode("utf-8", errors="replace")
            cycle = self._lib.verifier_error_cycle(self._handle)
            u = self._lib.verifier_error_location_u(self._handle)
            v = self._lib.verifier_error_location_v(self._handle)
            raise VerificationError(msg, cycle=cycle, location=(u, v))

    def evaluate(self) -> Metrics:
        """Run the solution and return its metrics."""
        cost = self._metric("cost")
        cycles = self._metric("cycles")
        area = self._metric("area")
        instructions = self._metric("instructions")
        return Metrics(cost=cost, cycles=cycles, area=area, instructions=instructions)

    def _metric(self, name: str) -> int:
        value = self._lib.verifier_evaluate_metric(self._handle, name.encode())
        if value == -1:
            self._check_error()
        return value

    def destroy(self) -> None:
        if self._handle:
            self._lib.verifier_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.destroy()

    def __enter__(self) -> Verifier:
        return self

    def __exit__(self, *args: object) -> None:
        self.destroy()


def verify(puzzle_bytes: bytes, solution_bytes: bytes, cycle_limit: int = 100_000) -> Metrics:
    """One-shot: verify a solution and return its metrics."""
    with Verifier(puzzle_bytes, solution_bytes, cycle_limit) as v:
        return v.evaluate()


def trace(
    puzzle_bytes: bytes,
    solution_bytes: bytes,
    max_cycles: int = 200,
    cycle_limit: int = 100_000,
) -> list[dict]:
    """Run a solution and return per-cycle state (atoms + arms) as a list of dicts.

    Each entry has: cycle, collision, atoms (list of [u,v,type,grabbed]),
    arms (list of [u,v,rotation,grabbing]).
    """
    import json

    lib = _get_lib()
    handle = lib.verifier_create_from_bytes(
        puzzle_bytes, len(puzzle_bytes),
        solution_bytes, len(solution_bytes),
    )
    if not handle:
        raise VerificationError("Failed to create verifier (null handle)")
    err = lib.verifier_error(handle)
    if err:
        msg = err.decode("utf-8", errors="replace")
        lib.verifier_destroy(handle)
        raise VerificationError(msg)
    if cycle_limit > 0:
        lib.verifier_set_cycle_limit(handle, cycle_limit)

    # Allocate a buffer for JSON output (16MB should be plenty)
    buf_size = 16 * 1024 * 1024
    buf = ctypes.create_string_buffer(buf_size)

    # Set up signature if not already done
    if not hasattr(lib, '_trace_setup'):
        lib.verifier_trace.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int
        ]
        lib.verifier_trace.restype = ctypes.c_int
        lib._trace_setup = True

    result = lib.verifier_trace(handle, max_cycles, buf, buf_size)
    if result < 0:
        err = lib.verifier_error(handle)
        msg = err.decode("utf-8", errors="replace") if err else "trace failed"
        lib.verifier_destroy(handle)
        raise VerificationError(msg)

    lib.verifier_destroy(handle)
    return json.loads(buf.value.decode("utf-8"))
