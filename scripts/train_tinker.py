#!/usr/bin/env python3
"""Train via Tinker's RL-as-a-service infrastructure.

This is the alternative to train.py (which runs on self-managed GPUs).
Tinker handles distributed training, GPU allocation, and optimization.

Requirements:
    pip install tinker-cookbook

Usage:
    python scripts/train_tinker.py
    python scripts/train_tinker.py model_name=Qwen/Qwen3-4B level=2
    python scripts/train_tinker.py use_structure_scoring=false  # ablation
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tinker")


@chz.chz
class Config:
    """Opus Magnum RL training config for Tinker."""

    # Model
    model_name: str = "Qwen/Qwen3.5-4B"

    # Training
    learning_rate: float = 4e-5
    max_tokens: int = 1024
    lora_rank: int = 32
    temperature: float = 0.7
    log_path: str = "/tmp/om-rl-tinker"
    eval_every: int = 10
    save_every: int = 25

    # Puzzles
    level: int = 1
    batch_size: int = 128
    group_size: int = 16
    num_puzzles: int = 1000
    seed: int = 42
    use_structure_scoring: bool = True
    campaign_puzzle_dir: str = "puzzles/campaign"

    # Optional
    wandb_project: str | None = None
    base_url: str | None = None


def main(config: Config) -> None:
    logger.info(f"Starting Tinker training:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Puzzle level: {config.level}")
    logger.info(f"  Batch: {config.batch_size}, Group: {config.group_size}")
    logger.info(f"  Max tokens: {config.max_tokens}")
    logger.info(f"  Structure scoring: {config.use_structure_scoring}")

    from om_rl.tinker.env import make_tinker_dataset_builder

    # Map model names to renderer names
    renderer_map = {
        "qwen3": "qwen3",
        "llama": "llama3",
        "deepseek": "deepseekv3",
    }
    renderer_name = "qwen3"  # default
    for key, name in renderer_map.items():
        if key in config.model_name.lower():
            renderer_name = name
            break

    dataset_builder = make_tinker_dataset_builder(
        complexity_level=config.level,
        batch_size=config.batch_size,
        group_size=config.group_size,
        num_puzzles=config.num_puzzles,
        campaign_puzzle_dir=config.campaign_puzzle_dir,
        seed=config.seed,
        use_structure_scoring=config.use_structure_scoring,
        model_name=config.model_name,
        renderer_name=renderer_name,
    )

    train_config = train.Config(
        model_name=config.model_name,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        log_path=config.log_path,
        eval_every=config.eval_every,
        save_every=config.save_every,
        wandb_project=config.wandb_project,
        base_url=config.base_url,
        remove_constant_reward_groups=True,
    )

    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
