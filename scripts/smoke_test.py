#!/usr/bin/env python3
"""Smoke test: exercise the full pipeline without a real model or GPU.

Uses a dummy "model" that generates random (mostly invalid) solutions
to verify that all components work together end-to-end:
- Puzzle generation + validation
- Environment reset/step
- Reward computation (including intermediate rewards)
- Rollout collection
- Dataset/curriculum management
- Difficulty evaluation

This should complete in seconds on CPU.
"""

import logging
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")


def random_solution(puzzle_name: str, rng: random.Random) -> str:
    """Generate a random (almost certainly invalid) solution."""
    arm_types = ["arm1", "arm2", "arm3", "piston"]
    glyph_types = [
        "bonder", "glyph-calcification", "glyph-projection", "glyph-disposal",
    ]
    instructions = ["G", "g", "R", "r", "E", "e", "P", "p"]

    lines = []
    # Random I/O placements
    lines.append(f"INPUT pos=({rng.randint(-5,5)},{rng.randint(-5,5)}) rot=0 idx=0")
    lines.append(f"OUTPUT pos=({rng.randint(-5,5)},{rng.randint(-5,5)}) rot=0 idx=0")

    # Random arm
    arm_type = rng.choice(arm_types)
    pos_u, pos_v = rng.randint(-3, 3), rng.randint(-3, 3)
    tape_entries = []
    for c in range(1, rng.randint(2, 8)):
        tape_entries.append(f"{c}:{rng.choice(instructions)}")
    tape = " ".join(tape_entries)
    lines.append(f"ARM {arm_type} pos=({pos_u},{pos_v}) rot=0 ext=1 id=0")
    lines.append(f"  TAPE: {tape}")

    # Maybe a glyph
    if rng.random() > 0.5:
        gt = rng.choice(glyph_types)
        lines.append(f"GLYPH {gt} pos=({rng.randint(-3,3)},{rng.randint(-3,3)}) rot=0")

    return "\n".join(lines)


def main():
    rng = random.Random(42)

    # ── 1. Puzzle Generation ────────────────────────────────────────────
    logger.info("=== 1. Puzzle Generation ===")
    from om_rl.puzzle_gen.generator import generate_puzzle, generate_puzzle_batch
    from om_rl.puzzle_gen.validator import validate_puzzle
    from vendor.opus_magnum.text_format import puzzle_to_text

    for level in range(1, 6):
        puzzles = generate_puzzle_batch(10, complexity_level=level, base_seed=level * 100)
        valid = sum(1 for p in puzzles if validate_puzzle(p))
        logger.info(f"  Level {level}: generated 10, valid {valid}")
    logger.info("  OK")

    # ── 2. Difficulty Evaluation ────────────────────────────────────────
    logger.info("=== 2. Difficulty Evaluation ===")
    from om_rl.complexity.evaluator import evaluate_difficulty

    for level in range(1, 6):
        p = generate_puzzle(complexity_level=level, seed=level * 10)
        diff = evaluate_difficulty(p)
        logger.info(f"  Level {level} ({p.name}): score={diff.score:.3f} "
                     f"(atoms={diff.atom_count}, bonds={diff.bond_count}, "
                     f"depth={diff.transformation_depth}, glyphs={diff.glyph_variety})")

    # Also test on a campaign puzzle
    from vendor.opus_magnum.puzzle_parser import parse_puzzle
    campaign = parse_puzzle("puzzles/campaign/P007.puzzle")
    diff = evaluate_difficulty(campaign)
    logger.info(f"  Campaign P007 ({campaign.name}): score={diff.score:.3f}")
    logger.info("  OK")

    # ── 3. Environment ──────────────────────────────────────────────────
    logger.info("=== 3. Environment (with intermediate rewards) ===")
    from om_rl.env.environment import OpusMagnumEnv, EnvironmentConfig
    from om_rl.env.reward import RewardConfig

    reward_cfg = RewardConfig(use_intermediate_rewards=True, intermediate_weight=0.3)
    env_cfg = EnvironmentConfig(max_attempts=3, reward_config=reward_cfg)
    env = OpusMagnumEnv(env_cfg)

    puzzle = generate_puzzle(complexity_level=1, seed=7)
    obs = env.reset(puzzle)
    logger.info(f"  Observation length: {len(obs)} chars")

    # Attempt 1: garbage
    result = env.step("not a solution", tokens_used=50)
    logger.info(f"  Attempt 1 (garbage): reward={result.reward:.3f}, done={result.done}")

    # Attempt 2: parseable but wrong
    sol_text = random_solution(puzzle.name, rng)
    result = env.step(sol_text, tokens_used=100)
    logger.info(f"  Attempt 2 (random): reward={result.reward:.3f}, done={result.done}, "
                 f"progress={result.info.get('progress_score', 'N/A')}")

    # Attempt 3: another random
    sol_text = random_solution(puzzle.name, rng)
    result = env.step(sol_text, tokens_used=150)
    logger.info(f"  Attempt 3 (random): reward={result.reward:.3f}, done={result.done}")
    logger.info("  OK")

    # ── 4. Rollout Collection (single-turn + multi-turn) ─────────────────
    logger.info("=== 4. Rollout Collection ===")
    from om_rl.training.rollout import collect_rollouts

    def dummy_generate(prompt: str) -> tuple[str, int]:
        sol = random_solution("dummy", rng)
        return sol, len(sol) // 4

    # Single-turn
    puzzles = [generate_puzzle(complexity_level=1, seed=i) for i in range(5)]
    batch = collect_rollouts(puzzles, dummy_generate, reward_cfg, max_attempts=1)
    stats = batch.stats()
    logger.info(f"  Single-turn: {stats['num_episodes']} episodes, "
                 f"solve={stats['solve_rate']:.2%}, "
                 f"attempts={stats['mean_attempts']:.1f}")

    # Multi-turn (3 attempts per puzzle)
    batch_mt = collect_rollouts(puzzles, dummy_generate, reward_cfg, max_attempts=3)
    stats_mt = batch_mt.stats()
    logger.info(f"  Multi-turn:  {stats_mt['num_episodes']} episodes, "
                 f"solve={stats_mt['solve_rate']:.2%}, "
                 f"attempts={stats_mt['mean_attempts']:.1f}, "
                 f"tokens={stats_mt['mean_tokens']:.0f}")

    # Verify multi-turn episodes have trajectories with feedback
    for ep in batch_mt.results[:2]:
        traj = ep.trajectory
        logger.info(f"    {ep.puzzle_name}: {ep.num_attempts} attempts, "
                     f"trajectory={len(traj)} chars, "
                     f"reward={ep.final_reward:.3f}")
    logger.info("  OK")

    # ── 5. Dataset + Curriculum ─────────────────────────────────────────
    logger.info("=== 5. Dataset + Curriculum ===")
    from om_rl.training.dataset import PuzzlePool
    from om_rl.training.config import CurriculumConfig

    pool = PuzzlePool()
    pool.load_campaign_puzzles("puzzles/campaign")
    pool.generate_puzzles(1, 50, base_seed=0)
    pool.generate_puzzles(2, 50, base_seed=1000)
    logger.info(f"  Total puzzles: {pool.total_puzzles}")
    logger.info(f"  Current level: {pool.current_level}")

    curriculum = CurriculumConfig(advance_threshold=0.3)
    sampled = pool.sample(10, curriculum)
    logger.info(f"  Sampled 10 puzzles from level {pool.current_level}")

    # Simulate curriculum advancement
    pool.maybe_advance_level(0.1, curriculum)  # Should not advance
    logger.info(f"  After 10% solve rate: level={pool.current_level}")
    pool.maybe_advance_level(0.35, curriculum)  # Should advance
    logger.info(f"  After 35% solve rate: level={pool.current_level}")
    logger.info("  OK")

    # ── 6. Mini Training Loop (multi-turn, dummy model, 3 steps) ────────
    logger.info("=== 6. Mini Training Loop (multi-turn, 3 steps) ===")
    from collections import defaultdict

    for step in range(1, 4):
        # Sample puzzles
        batch_puzzles = pool.sample(2, curriculum)

        # Generate K=2 multi-turn episodes per puzzle
        all_episodes = []
        for puzzle in batch_puzzles:
            for _ in range(2):
                rollout = collect_rollouts(
                    [puzzle], dummy_generate, reward_cfg, max_attempts=3
                )
                all_episodes.extend(rollout.results)

        # Compute GRPO advantages over episodes
        groups: dict[str, list] = defaultdict(list)
        for ep in all_episodes:
            groups[ep.puzzle_name].append(ep)

        advantages = []
        for name, group in groups.items():
            rewards = [ep.final_reward for ep in group]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
            std_r = var_r ** 0.5
            for ep in group:
                adv = (ep.final_reward - mean_r) / (std_r + 1e-8)
                advantages.append((ep, adv))

        mean_reward = sum(ep.final_reward for ep in all_episodes) / len(all_episodes)
        solve_rate = sum(1 for ep in all_episodes if ep.verified) / len(all_episodes)
        mean_attempts = sum(ep.num_attempts for ep in all_episodes) / len(all_episodes)
        logger.info(f"  Step {step}: {len(all_episodes)} episodes, "
                     f"reward={mean_reward:.3f}, solve={solve_rate:.2%}, "
                     f"attempts={mean_attempts:.1f}, "
                     f"advantages=[{', '.join(f'{a:.2f}' for _, a in advantages)}]")

    logger.info("  OK")

    # ── Done ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL SMOKE TESTS PASSED")
    logger.info("=" * 60)
    logger.info("")
    logger.info("The full pipeline works end-to-end. Next steps:")
    logger.info("  1. Set up a cloud GPU instance (A10G or L4)")
    logger.info("  2. pip install torch transformers trl peft bitsandbytes")
    logger.info("  3. python scripts/train.py --max-steps 100")
    logger.info("     (quick test to verify model loading + training loop)")
    logger.info("  4. python scripts/train.py --max-steps 1000")
    logger.info("     (first real training run on Level 1 puzzles)")


if __name__ == "__main__":
    main()
