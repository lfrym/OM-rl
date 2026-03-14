"""Client for the zlbb.faendir.com Opus Magnum leaderboard API.

Provides:
  - Puzzle metadata and file downloads
  - Human top scores (Pareto frontier) for comparison
  - Solution file downloads for reference
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


API_BASE = "https://zlbb.faendir.com/om"


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LeaderboardScore:
    cost: int
    cycles: int
    area: int
    instructions: int
    height: int | None = None
    width: float | None = None
    overlap: bool = False
    trackless: bool = False


@dataclass(slots=True)
class LeaderboardRecord:
    record_id: str
    puzzle_id: str
    score: LeaderboardScore
    category_ids: list[str] = field(default_factory=list)
    formatted: str = ""
    gif_url: str | None = None
    solution_url: str | None = None


@dataclass(slots=True)
class PuzzleInfo:
    puzzle_id: str
    display_name: str
    group_id: str
    group_name: str
    collection_id: str
    collection_name: str
    puzzle_type: str  # "normal" or "production"


# ── API helpers ─────────────────────────────────────────────────────────────


def _get_json(path: str) -> Any:
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get_bytes(path: str) -> bytes:
    url = f"{API_BASE}{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


# ── Public API ──────────────────────────────────────────────────────────────


def list_puzzles() -> list[PuzzleInfo]:
    """List all Opus Magnum puzzles from the leaderboard."""
    data = _get_json("/puzzles")
    results = []
    for p in data:
        group = p.get("group", {})
        collection = group.get("collection", {})
        results.append(
            PuzzleInfo(
                puzzle_id=p["id"],
                display_name=p.get("displayName", p["id"]),
                group_id=group.get("id", ""),
                group_name=group.get("displayName", ""),
                collection_id=collection.get("id", ""),
                collection_name=collection.get("displayName", ""),
                puzzle_type=p.get("type", "normal"),
            )
        )
    return results


def get_puzzle_info(puzzle_id: str) -> PuzzleInfo:
    """Get info for a single puzzle."""
    p = _get_json(f"/puzzle/{puzzle_id}")
    group = p.get("group", {})
    collection = group.get("collection", {})
    return PuzzleInfo(
        puzzle_id=p["id"],
        display_name=p.get("displayName", p["id"]),
        group_id=group.get("id", ""),
        group_name=group.get("displayName", ""),
        collection_id=collection.get("id", ""),
        collection_name=collection.get("displayName", ""),
        puzzle_type=p.get("type", "normal"),
    )


def download_puzzle(puzzle_id: str) -> bytes:
    """Download the binary .puzzle file for a given puzzle ID."""
    return _get_bytes(f"/puzzle/{puzzle_id}/file")


def download_puzzle_to(puzzle_id: str, dest: str | Path) -> Path:
    """Download a puzzle file and save it to disk."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = download_puzzle(puzzle_id)
    dest.write_bytes(data)
    return dest


def get_records(puzzle_id: str, include_frontier: bool = True) -> list[LeaderboardRecord]:
    """Get leaderboard records for a puzzle.

    With include_frontier=True, returns the full Pareto frontier of
    non-dominated solutions (the "best" scores along every trade-off).
    """
    frontier_param = "true" if include_frontier else "false"
    data = _get_json(f"/puzzle/{puzzle_id}/records?includeFrontier={frontier_param}")
    records = []
    for r in data:
        s = r.get("score", {})
        score = LeaderboardScore(
            cost=s.get("cost", 0),
            cycles=s.get("cycles", 0),
            area=s.get("area", 0),
            instructions=s.get("instructions", 0),
            height=s.get("height"),
            width=s.get("width"),
            overlap=s.get("overlap", False),
            trackless=s.get("trackless", False),
        )
        records.append(
            LeaderboardRecord(
                record_id=r.get("id", ""),
                puzzle_id=puzzle_id,
                score=score,
                category_ids=r.get("categoryIds", []),
                formatted=r.get("smartFormattedScore", ""),
                gif_url=r.get("gif"),
                solution_url=r.get("solution"),
            )
        )
    return records


def download_solution(puzzle_id: str, record_id: str) -> bytes:
    """Download the binary .solution file for a specific record."""
    return _get_bytes(f"/puzzle/{puzzle_id}/record/{record_id}/file")


def get_human_bests(puzzle_id: str) -> dict[str, LeaderboardScore]:
    """Get the best known human scores for a puzzle.

    Returns a dict with keys like "best_cost", "best_cycles", "best_area"
    pointing to the best score for each individual metric.
    """
    records = get_records(puzzle_id, include_frontier=True)
    if not records:
        return {}

    # Filter to non-overlap solutions for fair comparison.
    valid = [r for r in records if not r.score.overlap]
    if not valid:
        valid = records

    bests = {}
    bests["best_cost"] = min(valid, key=lambda r: r.score.cost).score
    bests["best_cycles"] = min(valid, key=lambda r: r.score.cycles).score
    bests["best_area"] = min(valid, key=lambda r: r.score.area).score
    bests["best_instructions"] = min(valid, key=lambda r: r.score.instructions).score

    # Also find the best "sum" (cost + cycles + area) as a balanced metric.
    bests["best_sum"] = min(
        valid,
        key=lambda r: r.score.cost + r.score.cycles + r.score.area,
    ).score

    return bests


def download_all_campaign_puzzles(dest_dir: str | Path) -> list[Path]:
    """Download all campaign (non-production) puzzle files."""
    dest_dir = Path(dest_dir)
    puzzles = list_puzzles()
    campaign = [p for p in puzzles if p.collection_id == "CAMPAIGN" and p.puzzle_type == "normal"]
    paths = []
    for p in campaign:
        group_dir = dest_dir / p.group_id.lower().replace("_", "-")
        group_dir.mkdir(parents=True, exist_ok=True)
        dest = group_dir / f"{p.puzzle_id}.puzzle"
        if not dest.exists():
            download_puzzle_to(p.puzzle_id, dest)
        paths.append(dest)
    return paths
