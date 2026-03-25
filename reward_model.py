"""Ensemble reward model and Peak-End preference loss for PbRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from env_utils import PreferenceSample


class RewardMLP(nn.Module):
    """Per-step reward predictor for low-dimensional state-action inputs."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.net(state_action).squeeze(-1)


class PeakEndBCE(nn.Module):
    """Binary cross-entropy over Bradley-Terry logits for preference learning."""

    def forward(
        self,
        utility_a: torch.Tensor,
        utility_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = utility_a - utility_b
        return F.binary_cross_entropy_with_logits(logits, labels)


class RewardMember(nn.Module):
    """Single ensemble member with learnable Peak-End weighting."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = RewardMLP(input_dim)
        self.omega_1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.omega_2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def _per_step_rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size, segment_len, _ = states.shape
        sa = torch.cat([states, actions], dim=-1).reshape(batch_size * segment_len, -1)
        rewards = self.model(sa).reshape(batch_size, segment_len)
        return rewards

    def segment_utility(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute utility with differentiable max operator over segment rewards."""
        rewards = self._per_step_rewards(states, actions)
        peak = torch.max(rewards, dim=1).values
        end = rewards[:, -1]
        return self.omega_1 * peak + self.omega_2 * end

    def step_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict per-step reward for single-step inputs [B, S], [B, A]."""
        sa = torch.cat([states, actions], dim=-1)
        return self.model(sa)


class PreferenceDataset(Dataset):
    """Torch dataset for pairwise trajectory preference supervision."""

    def __init__(self, samples: Sequence[PreferenceSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "states_a": torch.tensor(sample.segment_a.states, dtype=torch.float32),
            "actions_a": torch.tensor(sample.segment_a.actions, dtype=torch.float32),
            "states_b": torch.tensor(sample.segment_b.states, dtype=torch.float32),
            "actions_b": torch.tensor(sample.segment_b.actions, dtype=torch.float32),
            "label": torch.tensor(float(sample.label), dtype=torch.float32),
        }


@dataclass
class RewardTrainMetrics:
    loss: float
    pref_accuracy: float


class EnsembleRewardModel(nn.Module):
    """Ensemble of reward networks for preference learning and uncertainty estimation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 3,
        lr: float = 3e-4,
        device: str = "cpu",
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        input_dim = state_dim + action_dim

        members: List[RewardMember] = []
        for idx in range(ensemble_size):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed + idx)
                members.append(RewardMember(input_dim=input_dim))

        self.members = nn.ModuleList(members)
        self.loss_fn = PeakEndBCE()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    @torch.no_grad()
    def predict_step_ensemble(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-member predicted rewards with shape [M, B]."""
        preds = [member.step_reward(states, actions) for member in self.members]
        return torch.stack(preds, dim=0)

    @torch.no_grad()
    def sac_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        uncertainty_coef: float = 0.1,
    ) -> Tuple[float, float, float]:
        """Compute uncertainty-penalized reward r = mean - lambda * std."""
        states = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        pred_members = self.predict_step_ensemble(states, actions).squeeze(-1)
        mean_reward = pred_members.mean(dim=0)
        std_reward = pred_members.std(dim=0, unbiased=False)
        reward = mean_reward - uncertainty_coef * std_reward

        return float(reward.item()), float(mean_reward.item()), float(std_reward.item())

    def train_on_preferences(
        self,
        samples: Sequence[PreferenceSample],
        batch_size: int = 64,
        epochs: int = 1,
    ) -> RewardTrainMetrics:
        """Train ensemble on pairwise preferences using small VRAM-friendly batches."""
        if len(samples) == 0:
            return RewardTrainMetrics(loss=0.0, pref_accuracy=0.0)

        dataset = PreferenceDataset(samples)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        self.train()
        total_loss = 0.0
        total_count = 0
        correct = 0

        for _ in range(epochs):
            for batch in loader:
                states_a = batch["states_a"].to(self.device)
                actions_a = batch["actions_a"].to(self.device)
                states_b = batch["states_b"].to(self.device)
                actions_b = batch["actions_b"].to(self.device)
                labels = batch["label"].to(self.device)

                losses = []
                logits_for_acc = []
                for member in self.members:
                    utility_a = member.segment_utility(states_a, actions_a)
                    utility_b = member.segment_utility(states_b, actions_b)
                    losses.append(self.loss_fn(utility_a, utility_b, labels))
                    logits_for_acc.append((utility_a - utility_b).detach())

                loss = torch.stack(losses).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    ensemble_logits = torch.stack(logits_for_acc, dim=0).mean(dim=0)
                    pred = (torch.sigmoid(ensemble_logits) > 0.5).float()
                    correct += int((pred == labels).sum().item())
                    batch_count = labels.numel()
                    total_count += batch_count
                    total_loss += float(loss.item()) * batch_count

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(total_count, 1)
        acc = correct / max(total_count, 1)
        return RewardTrainMetrics(loss=avg_loss, pref_accuracy=acc)

    @torch.no_grad()
    def estimate_ensemble_variance(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Estimate mean variance over a batch of state-action inputs."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        preds = self.predict_step_ensemble(states, actions)
        variance = preds.var(dim=0, unbiased=False).mean()
        return float(variance.item())
