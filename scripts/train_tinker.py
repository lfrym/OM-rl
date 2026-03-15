#!/usr/bin/env python3
"""Train via Tinker's RL-as-a-service infrastructure.

This is the alternative to train.py (which runs on self-managed GPUs).
Tinker handles distributed training, GPU allocation, and optimization.

Requirements:
    pip install tinker-cookbook

Usage:
    # Basic run (Qwen3-4B, Level 1 puzzles)
    python scripts/train_tinker.py

    # Custom model and settings
    python scripts/train_tinker.py --model Qwen/Qwen3-4B --level 2

    # Pass additional Tinker config via chz CLI syntax
    python scripts/train_tinker.py --max_tokens 1024 --learning_rate 4e-5
"""

import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tinker")


@chz.chz
class Config:
    """Opus Magnum RL training config for Tinker."""

    # Tinker training settings
    model_name: str = "Qwen/Qwen3-4B"
    learning_rate: float = 4e-5
    max_tokens: int = 1024
    lora_rank: int = 32
    temperature: float = 0.7
    log_path: str = "/tmp/om-rl-tinker"
    max_steps: int = 100
    eval_every: int = 10
    save_every: int = 25
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None

    # Puzzle settings
    level: int = 1
    batch_size: int = 128
    group_size: int = 16
    num_puzzles: int = 1000
    seed: int = 42
    use_structure_scoring: bool = True


async def main(config: Config) -> None:
    logger.info(f"Starting Tinker training:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Puzzle level: {config.level}")
    logger.info(f"  Batch size: {config.batch_size}, Group size: {config.group_size}")
    logger.info(f"  Max tokens: {config.max_tokens}")
    logger.info(f"  Structure scoring: {config.use_structure_scoring}")
    logger.info(f"  Log dir: {config.log_path}")

    dataset_builder = OpusMagnumDatasetBuilder(
        complexity_level=config.level,
        batch_size=config.batch_size,
        group_size=config.group_size,
        num_puzzles=config.num_puzzles,
        seed=config.seed,
        use_structure_scoring=config.use_structure_scoring,
    )

    train_config = train.Config(
        model_name=config.model_name,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        log_path=config.log_path,
        max_steps=config.max_steps,
        eval_every=config.eval_every,
        save_every=config.save_every,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        base_url=config.base_url,
        remove_constant_reward_groups=True,
    )

    await train.main(train_config)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
