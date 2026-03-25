"""Main training loop for memory-constrained PbRL with Peak-End preferences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env_utils import PreferenceSample, SegmentExtractor, make_env, sample_segment_pairs, set_global_seeds
from oracle import SyntheticOracle
from reward_model import EnsembleRewardModel
from sac_agent import SACAgent, SACConfig


@dataclass
class TrainConfig:
    seed: int = 42
    total_steps: int = 300_000
    start_steps: int = 5_000
    batch_size: int = 256
    update_after: int = 1_000
    update_every: int = 1

    segment_length: int = 30
    preference_pairs_per_update: int = 200
    preference_update_interval: int = 5_000
    reward_batch_size: int = 64
    reward_epochs: int = 3

    ensemble_size: int = 3
    uncertainty_coef: float = 0.1
    env_name: str = "auto"

    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5_000
    resume_path: str = ""

    log_dir: str = "runs/pbrl_peak_end"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainConfig:
    """Parse command-line options into a TrainConfig object."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--log-dir", type=str, default="runs/pbrl_peak_end")
    parser.add_argument("--env-name", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--resume-path", type=str, default="")

    args = parser.parse_args()
    config = TrainConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        device=args.device,
        log_dir=args.log_dir,
        env_name=args.env_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_path=args.resume_path,
    )
    return config


def _save_checkpoint(
    save_path: Path,
    step: int,
    episode_idx: int,
    sac: SACAgent,
    reward_model: EnsembleRewardModel,
    all_segments: List[Any],
    preference_buffer: List[PreferenceSample],
) -> None:
    """Persist training state so long runs can resume after interruption."""
    payload: Dict[str, Any] = {
        "step": step,
        "episode_idx": episode_idx,
        "sac": sac.state_dict(),
        "reward_model": reward_model.state_dict(),
        "reward_model_optimizer": reward_model.optimizer.state_dict(),
        "all_segments": all_segments,
        "preference_buffer": preference_buffer,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)


def _load_checkpoint(
    load_path: Path,
    sac: SACAgent,
    reward_model: EnsembleRewardModel,
) -> Dict[str, Any]:
    """Restore training state from checkpoint file."""
    # This checkpoint is created by this project and intentionally stores
    # non-tensor Python objects (e.g., replay buffer numpy arrays, buffers).
    # PyTorch 2.6+ defaults torch.load(..., weights_only=True), which rejects
    # these objects. For trusted local checkpoints we must opt out explicitly.
    payload = torch.load(load_path, map_location=reward_model.device, weights_only=False)
    sac.load_state_dict(payload["sac"])
    reward_model.load_state_dict(payload["reward_model"])
    reward_model.optimizer.load_state_dict(payload["reward_model_optimizer"])
    return payload


def train(config: TrainConfig) -> None:
    """Run active preference collection, reward model fitting, and SAC learning."""
    set_global_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    env = make_env(env_name=config.env_name, seed=config.seed)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    action_limit = float(env.action_space.high[0])

    sac = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_limit=action_limit,
        device=config.device,
        config=SACConfig(),
    )
    reward_model = EnsembleRewardModel(
        state_dim=state_dim,
        action_dim=action_dim,
        ensemble_size=config.ensemble_size,
        device=config.device,
        base_seed=config.seed,
    )
    oracle = SyntheticOracle(noise_prob=0.10, seed=config.seed)
    segment_extractor = SegmentExtractor(segment_length=config.segment_length)

    all_segments: List[Any] = []
    preference_buffer: List[PreferenceSample] = []

    writer = SummaryWriter(log_dir=config.log_dir)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    state, _ = env.reset(seed=config.seed)
    episode_true_return = 0.0
    episode_length = 0
    episode_idx = 0

    start_step = 1
    if config.resume_path:
        ckpt_path = Path(config.resume_path)
        if ckpt_path.exists():
            restored = _load_checkpoint(ckpt_path, sac=sac, reward_model=reward_model)
            all_segments = list(restored.get("all_segments", []))
            preference_buffer = list(restored.get("preference_buffer", []))
            episode_idx = int(restored.get("episode_idx", 0))
            start_step = int(restored.get("step", 0)) + 1
            print(f"Resumed from {ckpt_path} at step={start_step}.")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    for step in range(start_step, config.total_steps + 1):
        if step <= config.start_steps:
            action = env.action_space.sample()
        else:
            action = sac.select_action(state, deterministic=False)

        next_state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        true_reward = float(info["true_reward"])
        episode_true_return += true_reward
        episode_length += 1

        segment = segment_extractor.add_step(state=state, action=action, true_reward=true_reward)
        if segment is not None:
            all_segments.append(segment)

        learned_reward, mean_pred, std_pred = reward_model.sac_reward(
            state=state,
            action=action,
            uncertainty_coef=config.uncertainty_coef,
        )

        sac.replay_buffer.add(
            state=np.asarray(state, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=learned_reward,
            next_state=np.asarray(next_state, dtype=np.float32),
            done=done,
        )

        if step >= config.update_after and step % config.update_every == 0:
            update_metrics = sac.update(batch_size=config.batch_size)
            writer.add_scalar("sac/critic_loss", update_metrics["critic_loss"], global_step=step)
            writer.add_scalar("sac/actor_loss", update_metrics["actor_loss"], global_step=step)
            writer.add_scalar("sac/alpha", update_metrics["alpha"], global_step=step)

        writer.add_scalar("reward_model/step_mean_prediction", mean_pred, global_step=step)
        writer.add_scalar("reward_model/step_std_prediction", std_pred, global_step=step)
        writer.add_scalar("reward_model/step_variance", std_pred**2, global_step=step)

        if step % config.preference_update_interval == 0 and len(all_segments) >= 2:
            pairs = sample_segment_pairs(
                segments=all_segments,
                n_pairs=config.preference_pairs_per_update,
                rng=rng,
            )
            new_samples = oracle.build_preference_samples(pairs)
            preference_buffer.extend(new_samples)

            metrics = reward_model.train_on_preferences(
                samples=preference_buffer,
                batch_size=config.reward_batch_size,
                epochs=config.reward_epochs,
            )

            writer.add_scalar("reward_model/preference_loss", metrics.loss, global_step=step)
            writer.add_scalar("reward_model/preference_accuracy", metrics.pref_accuracy, global_step=step)

            if len(preference_buffer) > 0:
                probe = preference_buffer[-1].segment_a
                probe_states = torch.tensor(probe.states, dtype=torch.float32)
                probe_actions = torch.tensor(probe.actions, dtype=torch.float32)
                variance = reward_model.estimate_ensemble_variance(probe_states, probe_actions)
                writer.add_scalar("reward_model/ensemble_variance_probe", variance, global_step=step)

            print(
                f"[Step {step}] pref_samples={len(preference_buffer)} "
                f"loss={metrics.loss:.4f} acc={metrics.pref_accuracy:.3f}"
            )

        if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
            ckpt_path = Path(config.checkpoint_dir) / f"pbrl_step_{step}.pt"
            _save_checkpoint(
                save_path=ckpt_path,
                step=step,
                episode_idx=episode_idx,
                sac=sac,
                reward_model=reward_model,
                all_segments=all_segments,
                preference_buffer=preference_buffer,
            )
            print(f"Saved checkpoint to {ckpt_path}")

        state = next_state

        if done:
            writer.add_scalar("env/true_return", episode_true_return, global_step=step)
            writer.add_scalar("env/episode_length", episode_length, global_step=step)
            print(
                f"[Episode {episode_idx}] step={step} "
                f"true_return={episode_true_return:.2f} length={episode_length}"
            )

            state, _ = env.reset()
            segment_extractor.reset()
            episode_true_return = 0.0
            episode_length = 0
            episode_idx += 1

    writer.close()
    env.close()


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
