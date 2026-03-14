"""GRPO trainer for Opus Magnum RL.

Custom GRPO training loop that supports multi-turn episodes where the model
submits solutions, sees error feedback from omsim, and iterates. The full
trajectory (all attempts + feedback) is treated as the "completion" for
GRPO's policy gradient.

The reward signal comes from omsim verification + token efficiency +
intermediate progress scoring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .dataset import PuzzlePool
from .rollout import collect_rollouts, RolloutBatch, EpisodeResult

logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer with optional QLoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if config.model.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        **model_kwargs,
    )

    if config.model.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if config.model.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train(config: TrainingConfig) -> None:
    """Main training loop.

    GRPO with multi-turn episodes:
    1. Sample puzzles from curriculum
    2. For each puzzle, run K multi-turn episodes (model submits, sees feedback, iterates)
    3. Compute rewards: best outcome across all attempts in each episode
    4. Compute group-relative advantages (group = K episodes on same puzzle)
    5. Update policy with REINFORCE-style gradient over full trajectories
    """
    import torch
    from torch.optim import AdamW
    import json
    import time

    logger.info(f"Starting training with config: {config}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "model_name": config.model.model_name,
            "use_lora": config.model.use_lora,
            "learning_rate": config.learning_rate,
            "num_completions": config.num_completions,
            "batch_size": config.batch_size,
            "max_attempts": config.max_attempts,
            "kl_coeff": config.kl_coeff,
            "intermediate_rewards": config.reward.use_intermediate_rewards,
        }, f, indent=2)

    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    # Load puzzle pool
    pool = PuzzlePool()
    pool.load_campaign_puzzles(config.puzzle_dir)
    pool.generate_puzzles(
        level=1,
        count=config.curriculum.puzzles_per_level,
        base_seed=config.seed,
    )

    # Training loop
    step = 0
    epoch_stats: list[dict] = []

    def generate_fn(prompt: str) -> tuple[str, int]:
        """Generate a completion using the model.

        In multi-turn mode, the prompt grows with each turn (includes prior
        attempts and feedback), so the model sees its previous mistakes.
        """
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the generated part
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        tokens_used = len(gen_ids)
        return completion, tokens_used

    logger.info(f"Starting training loop (max_steps={config.max_steps}, "
                f"max_attempts={config.max_attempts})")

    while step < config.max_steps:
        step_start = time.time()

        # Sample puzzles
        puzzles = pool.sample(config.batch_size, config.curriculum)

        # Collect rollouts: K multi-turn episodes per puzzle
        all_episodes: list[EpisodeResult] = []
        for puzzle in puzzles:
            for _ in range(config.num_completions):
                batch = collect_rollouts(
                    [puzzle], generate_fn, config.reward, config.cycle_limit,
                    max_attempts=config.max_attempts,
                )
                all_episodes.extend(batch.results)

        # Compute group-relative advantages
        # Group by puzzle, then normalize rewards within each group
        from collections import defaultdict
        groups: dict[str, list[EpisodeResult]] = defaultdict(list)
        for ep in all_episodes:
            groups[ep.puzzle_name].append(ep)

        # GRPO: advantage = (reward - group_mean) / (group_std + eps)
        advantages: list[tuple[EpisodeResult, float]] = []
        for puzzle_name, group in groups.items():
            rewards = [ep.final_reward for ep in group]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
            std_r = var_r ** 0.5
            for ep in group:
                adv = (ep.final_reward - mean_r) / (std_r + 1e-8)
                advantages.append((ep, adv))

        # Compute policy gradient loss over full trajectories
        total_loss = torch.tensor(0.0, device=model.device)
        num_tokens = 0

        for episode, advantage in advantages:
            trajectory = episode.trajectory
            if not trajectory.strip():
                continue

            # Tokenize prompt + full trajectory
            full_text = episode.prompt + trajectory
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                               max_length=2048 + config.model.max_new_tokens * config.max_attempts
                               ).to(model.device)

            prompt_ids = tokenizer(episode.prompt, return_tensors="pt",
                                   truncation=True, max_length=2048)["input_ids"]
            prompt_len = prompt_ids.shape[1]

            if inputs["input_ids"].shape[1] <= prompt_len:
                continue

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            # Get per-token log probs for the trajectory part only
            logits = outputs.logits[:, prompt_len - 1:-1, :]
            labels = inputs["input_ids"][:, prompt_len:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            # REINFORCE loss: -advantage * sum(log_probs)
            trajectory_log_prob = token_log_probs.sum()
            total_loss -= advantage * trajectory_log_prob
            num_tokens += labels.shape[1]

        if num_tokens > 0:
            total_loss = total_loss / len(advantages)

            # Backward + update
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1
        step_time = time.time() - step_start

        # Logging
        batch_stats = {
            "step": step,
            "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else 0,
            "mean_reward": sum(ep.final_reward for ep in all_episodes) / max(len(all_episodes), 1),
            "solve_rate": sum(1 for ep in all_episodes if ep.verified) / max(len(all_episodes), 1),
            "num_verified": sum(1 for ep in all_episodes if ep.verified),
            "total_episodes": len(all_episodes),
            "mean_attempts": sum(ep.num_attempts for ep in all_episodes) / max(len(all_episodes), 1),
            "mean_tokens": sum(ep.total_tokens for ep in all_episodes) / max(len(all_episodes), 1),
            "curriculum_level": pool.current_level,
            "step_time": step_time,
        }
        epoch_stats.append(batch_stats)

        if step % config.log_every == 0:
            logger.info(
                f"Step {step}: loss={batch_stats['loss']:.4f} "
                f"reward={batch_stats['mean_reward']:.3f} "
                f"solve={batch_stats['solve_rate']:.2%} "
                f"attempts={batch_stats['mean_attempts']:.1f} "
                f"level={pool.current_level} "
                f"time={step_time:.1f}s"
            )

        # Evaluation (multi-turn eval too)
        if step % config.eval_every == 0:
            eval_puzzles = pool.sample(config.eval_puzzles, config.curriculum)
            eval_batch = collect_rollouts(
                eval_puzzles, generate_fn, config.reward, config.cycle_limit,
                max_attempts=config.max_attempts,
            )
            eval_stats = eval_batch.stats()
            logger.info(
                f"Eval at step {step}: "
                f"solve_rate={eval_stats['solve_rate']:.2%} "
                f"mean_reward={eval_stats['mean_reward']:.3f} "
                f"mean_attempts={eval_stats['mean_attempts']:.1f}"
            )

            # Curriculum advancement
            pool.maybe_advance_level(eval_stats["solve_rate"], config.curriculum)

            # Generate puzzles for new level if needed
            if pool.current_level not in pool.generated_puzzles:
                pool.generate_puzzles(
                    pool.current_level,
                    config.curriculum.puzzles_per_level,
                    base_seed=config.seed + step,
                )

        # Checkpointing
        if step % config.checkpoint_every == 0:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint to {ckpt_dir}")

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training stats
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(epoch_stats, f, indent=2)

    logger.info(f"Training complete. Final model saved to {final_dir}")
