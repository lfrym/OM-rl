#!/usr/bin/env python3
"""Train via Tinker's RL-as-a-service infrastructure.

This is the alternative to train.py (which runs on self-managed GPUs).
Tinker handles distributed training, GPU allocation, and optimization.

Requirements:
    pip install tinker-cookbook

Usage:
    python scripts/train_tinker.py
    python scripts/train_tinker.py model_name=Qwen/Qwen3-4B-Instruct-2507 level=2
    python scripts/train_tinker.py use_structure_scoring=false  # ablation
"""

import asyncio
import logging
import sys
import time
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


def _next_log_path(base: str = "outputs/tinker") -> str:
    """Find the next available run directory (run_001, run_002, ...)."""
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        candidate = base_path / f"run_{i:03d}"
        if not candidate.exists():
            return str(candidate)
    return str(base_path / f"run_{int(time.time())}")


@chz.chz
class Config:
    """Opus Magnum RL training config for Tinker."""

    # Model
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Training
    learning_rate: float = 4e-5
    max_tokens: int = 8192
    max_steps: int = 30
    lora_rank: int = 32
    temperature: float = 0.7
    kl_penalty_coef: float = 0.1
    eval_every: int = 10
    save_every: int = 10

    # Puzzles
    level: int = 1
    max_level: int = 3
    curriculum_step_interval: int = 10
    batch_size: int = 4
    group_size: int = 4
    num_puzzles: int = 1000
    seed: int = 42
    use_structure_scoring: bool = True

    # Multi-turn
    max_attempts: int = 5

    # Logging & overrides
    wandb_project: str | None = "om-rl"
    base_url: str | None = None
    renderer_name: str | None = None  # Auto-detected from model_name if None


def main(config: Config) -> None:
    log_path = _next_log_path()

    logger.info(f"Starting Tinker training:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Puzzle levels: {config.level}-{config.max_level} "
                f"(advance every {config.curriculum_step_interval} steps)")
    logger.info(f"  Batch: {config.batch_size}, Group: {config.group_size}")
    logger.info(f"  Max tokens: {config.max_tokens}, Max steps: {config.max_steps}")
    logger.info(f"  KL penalty: {config.kl_penalty_coef}")
    logger.info(f"  Structure scoring: {config.use_structure_scoring}")
    logger.info(f"  Max attempts per puzzle: {config.max_attempts}")
    logger.info(f"  Log path: {log_path}")

    from om_rl.tinker.env import make_tinker_dataset_builder

    # Determine renderer
    if config.renderer_name:
        renderer_name = config.renderer_name
    else:
        renderer_map = {
            "qwen3": "qwen3",
            "llama": "llama3",
            "deepseek": "deepseekv3",
            "gpt-oss": "gpt_oss_no_sysprompt",
        }
        renderer_name = "qwen3"  # default
        for key, name in renderer_map.items():
            if key in config.model_name.lower():
                renderer_name = name
                break

    dataset_builder = make_tinker_dataset_builder(
        complexity_level=config.level,
        max_level=config.max_level,
        curriculum_step_interval=config.curriculum_step_interval,
        batch_size=config.batch_size,
        group_size=config.group_size,
        max_steps=config.max_steps,
        num_puzzles=config.num_puzzles,
        seed=config.seed,
        use_structure_scoring=config.use_structure_scoring,
        model_name=config.model_name,
        renderer_name=renderer_name,
        max_tokens=config.max_tokens,
        max_attempts=config.max_attempts,
    )

    train_config = train.Config(
        model_name=config.model_name,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        kl_penalty_coef=config.kl_penalty_coef,
        log_path=log_path,
        eval_every=config.eval_every,
        save_every=config.save_every,
        wandb_project=config.wandb_project,
        base_url=config.base_url,
        remove_constant_reward_groups=True,
    )

    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
