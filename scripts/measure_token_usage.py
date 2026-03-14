#!/usr/bin/env python3
"""Measure how many tokens the model actually generates at various limits.

Runs inference on a few puzzles with increasing max_new_tokens to find
the natural completion length. No training — just generation.

Usage:
    python scripts/measure_token_usage.py [--model MODEL] [--num-puzzles N]
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("measure")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-puzzles", type=int, default=3)
    parser.add_argument("--limits", default="256,512,1024,2048,4096,8192",
                        help="Comma-separated max_new_tokens values to test")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    limits = [int(x) for x in args.limits.split(",")]

    # Load model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    logger.info("Model loaded.")

    # Generate puzzles
    from om_rl.puzzle_gen.generator import generate_puzzle
    from om_rl.env.observation import format_initial_observation

    puzzles = [generate_puzzle(complexity_level=1, seed=s) for s in [1, 10, 63]][:args.num_puzzles]

    # Test each limit
    for limit in limits:
        logger.info(f"\n{'='*60}")
        logger.info(f"max_new_tokens = {limit}")
        logger.info(f"{'='*60}")

        gen_lengths = []
        hit_limit = 0

        for puzzle in puzzles:
            prompt = format_initial_observation(puzzle)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=2048).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            start = time.monotonic()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=limit,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.monotonic() - start

            gen_ids = outputs[0][prompt_len:]
            gen_len = len(gen_ids)
            tok_per_sec = gen_len / elapsed if elapsed > 0 else 0
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Check if it hit the limit (didn't stop naturally)
            reached_limit = gen_len >= limit - 1  # -1 for potential off-by-one
            if reached_limit:
                hit_limit += 1

            gen_lengths.append(gen_len)

            # Show preview
            preview = gen_text[:200].replace('\n', '\\n')
            ending = gen_text[-100:].replace('\n', '\\n') if len(gen_text) > 200 else ""

            logger.info(f"  {puzzle.name}: {gen_len} tokens in {elapsed:.1f}s ({tok_per_sec:.0f} tok/s)"
                        f" {'** HIT LIMIT **' if reached_limit else '(stopped naturally)'}")
            logger.info(f"    Start: {preview}")
            if ending:
                logger.info(f"    End:   ...{ending}")

        avg = sum(gen_lengths) / len(gen_lengths)
        logger.info(f"\n  Summary: avg={avg:.0f} tokens, "
                    f"hit_limit={hit_limit}/{len(puzzles)}, "
                    f"range={min(gen_lengths)}-{max(gen_lengths)}")

        # If no generation hit the limit, we've found the natural length
        if hit_limit == 0:
            logger.info(f"\n  >>> No generations hit the limit at {limit}. Natural completion length found.")
            logger.info(f"  >>> Recommended max_new_tokens: {int(max(gen_lengths) * 1.5)} (1.5x max observed)")
            break

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
