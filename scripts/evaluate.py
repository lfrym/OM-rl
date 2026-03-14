#!/usr/bin/env python3
"""Evaluate a model on Opus Magnum puzzles."""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from om_rl.training.config import TrainingConfig, ModelConfig
from om_rl.training.dataset import PuzzlePool
from om_rl.training.eval import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on Opus Magnum puzzles")
    parser.add_argument("--model", required=True, help="Model name or checkpoint path")
    parser.add_argument("--puzzle-dir", default="puzzles/campaign", help="Puzzle directory")
    parser.add_argument("--num-puzzles", type=int, default=None, help="Number of puzzles (all if None)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--generated-level", type=int, default=None, help="Evaluate on generated puzzles at this level")
    parser.add_argument("--generated-count", type=int, default=50, help="Number of generated puzzles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load puzzles
    pool = PuzzlePool()
    pool.load_campaign_puzzles(args.puzzle_dir)

    if args.generated_level:
        pool.generate_puzzles(args.generated_level, args.generated_count)
        puzzles = pool.generated_puzzles[args.generated_level]
    else:
        puzzles = pool.campaign_puzzles

    if args.num_puzzles:
        puzzles = puzzles[: args.num_puzzles]

    print(f"Evaluating on {len(puzzles)} puzzles...")

    # Load model
    from om_rl.training.trainer import setup_model_and_tokenizer
    import torch

    config = TrainingConfig(model=ModelConfig(model_name=args.model))
    model, tokenizer = setup_model_and_tokenizer(config)

    def generate_fn(prompt: str) -> tuple[str, int]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True), len(gen_ids)

    result = evaluate(puzzles, generate_fn)
    print(result.summary())

    if args.output:
        result.to_json(args.output)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
