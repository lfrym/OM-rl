"""Training configuration for Opus Magnum RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from om_rl.env.reward import RewardConfig


@dataclass
class ModelConfig:
    """Model and LoRA configuration."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    load_in_4bit: bool = True  # QLoRA
    max_new_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    # Complexity levels to train on (in order)
    levels: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    # Advance to next level when solve rate exceeds this threshold
    advance_threshold: float = 0.3
    # Fraction of generated vs campaign puzzles
    generated_ratio: float = 0.7
    # Number of puzzles to generate per level per epoch
    puzzles_per_level: int = 100


@dataclass
class TrainingConfig:
    """Full training configuration."""

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # GRPO parameters
    num_completions: int = 4  # K completions per puzzle
    learning_rate: float = 5e-6
    kl_coeff: float = 0.05
    batch_size: int = 4  # Puzzles per batch (each generates K completions)
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: int = 10_000

    # Environment
    max_attempts: int = 3  # Multi-turn: model submits, sees feedback, iterates
    cycle_limit: int = 100_000
    reward: RewardConfig = field(default_factory=lambda: RewardConfig(use_intermediate_rewards=True))

    # Curriculum
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Paths
    output_dir: str = "outputs"
    puzzle_dir: str = "puzzles/campaign"
    checkpoint_every: int = 500

    # Logging
    log_every: int = 10
    eval_every: int = 100
    eval_puzzles: int = 20  # Number of puzzles to evaluate on

    # Misc
    seed: int = 42
