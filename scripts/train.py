#!/usr/bin/env python3
"""Main training entry point for Opus Magnum RL."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from om_rl.training.config import TrainingConfig, ModelConfig, CurriculumConfig
from om_rl.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train an RL model to solve Opus Magnum puzzles")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Model name or path")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--puzzle-dir", default="puzzles/campaign", help="Campaign puzzle directory")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Puzzles per batch")
    parser.add_argument("--num-completions", type=int, default=4, help="Completions per puzzle (K)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full finetune)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max submissions per episode (multi-turn)")
    parser.add_argument("--start-level", type=int, default=1, help="Starting curriculum level")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TrainingConfig(
        model=ModelConfig(
            model_name=args.model,
            use_lora=not args.no_lora,
            load_in_4bit=not args.no_4bit,
        ),
        num_completions=args.num_completions,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_attempts=args.max_attempts,
        output_dir=args.output_dir,
        puzzle_dir=args.puzzle_dir,
        seed=args.seed,
    )

    config.curriculum.levels = list(range(args.start_level, 6))

    logging.info(f"Training config: model={config.model.model_name}, "
                 f"lora={config.model.use_lora}, 4bit={config.model.load_in_4bit}, "
                 f"max_attempts={config.max_attempts}")

    train(config)


if __name__ == "__main__":
    main()
