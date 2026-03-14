#!/usr/bin/env python3
"""Train via Tinker's RL-as-a-service infrastructure.

This is the alternative to train.py (which runs on self-managed GPUs).
Tinker handles distributed training, GPU allocation, and optimization.

Requirements:
    pip install tinker-cookbook

Usage:
    # Basic run (Qwen3.5-4B, Level 1 puzzles)
    python scripts/train_tinker.py

    # Custom model and settings
    python scripts/train_tinker.py --model Qwen/Qwen3-4B --level 2 --max-tokens 1024

    # Disable structure scoring (ablation)
    python scripts/train_tinker.py --no-structure-scoring
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Train on Opus Magnum via Tinker")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Model to train (must be supported by Tinker)")
    parser.add_argument("--level", type=int, default=1,
                        help="Puzzle complexity level (1-5)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Puzzles per batch")
    parser.add_argument("--group-size", type=int, default=16,
                        help="Completions per puzzle (K for GRPO)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens per generation")
    parser.add_argument("--lr", type=float, default=4e-5,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--num-puzzles", type=int, default=1000,
                        help="Total puzzles in training pool")
    parser.add_argument("--log-dir", default="/tmp/om-rl-tinker",
                        help="Log directory")
    parser.add_argument("--no-structure-scoring", action="store_true",
                        help="Disable structure scoring (ablation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate every N steps (0=disabled)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save checkpoint every N steps (0=disabled)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (None=disabled)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from tinker_cookbook.rl import train
        import chz
    except ImportError:
        print("Error: tinker-cookbook is not installed.")
        print("Install it with: pip install tinker-cookbook")
        print("")
        print("For self-managed GPU training, use: python scripts/train.py")
        sys.exit(1)

    from om_rl.tinker.env import OpusMagnumDatasetBuilder

    dataset_builder = OpusMagnumDatasetBuilder(
        complexity_level=args.level,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_puzzles=args.num_puzzles,
        seed=args.seed,
        use_structure_scoring=not args.no_structure_scoring,
    )

    # Build Tinker config
    config = chz.Blueprint(train.Config).create(
        model_name=args.model,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,
        lora_rank=args.lora_rank,
        log_path=args.log_dir,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_steps=args.max_steps,
        temperature=0.7,
        wandb_project=args.wandb_project,
        # Filter out groups where all completions get the same reward
        # (same issue we had with our custom GRPO loop)
        remove_constant_reward_groups=True,
    )

    # Safety check for log dir
    log_path = Path(args.log_dir)
    if log_path.exists():
        logging.warning(f"Log directory {log_path} already exists")

    logging.info(f"Starting Tinker training:")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Puzzle level: {args.level}")
    logging.info(f"  Batch size: {args.batch_size}, Group size: {args.group_size}")
    logging.info(f"  Max tokens: {args.max_tokens}")
    logging.info(f"  Structure scoring: {not args.no_structure_scoring}")
    logging.info(f"  Log dir: {args.log_dir}")

    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
