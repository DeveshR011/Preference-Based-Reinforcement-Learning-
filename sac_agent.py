"""Soft Actor-Critic agent for continuous control with learned reward input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """Simple replay buffer for off-policy continuous control."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "states": torch.tensor(self.states[idx], device=self.device),
            "actions": torch.tensor(self.actions[idx], device=self.device),
            "rewards": torch.tensor(self.rewards[idx], device=self.device),
            "next_states": torch.tensor(self.next_states[idx], device=self.device),
            "dones": torch.tensor(self.dones[idx], device=self.device),
        }
        return batch

    def state_dict(self) -> Dict[str, np.ndarray | int]:
        """Return serializable replay buffer state."""
        return {
            "capacity": self.capacity,
            "ptr": self.ptr,
            "size": self.size,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
        }

    def load_state_dict(self, state: Dict[str, np.ndarray | int]) -> None:
        """Restore replay buffer state from checkpoint."""
        self.ptr = int(state["ptr"])
        self.size = int(state["size"])
        self.states[...] = np.asarray(state["states"], dtype=np.float32)
        self.actions[...] = np.asarray(state["actions"], dtype=np.float32)
        self.rewards[...] = np.asarray(state["rewards"], dtype=np.float32)
        self.next_states[...] = np.asarray(state["next_states"], dtype=np.float32)
        self.dones[...] = np.asarray(state["dones"], dtype=np.float32)


class MLP(nn.Module):
    """Generic two-layer MLP block."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    """Gaussian policy with tanh-squashed actions."""

    def __init__(self, state_dim: int, action_dim: int, action_limit: float) -> None:
        super().__init__()
        self.backbone = MLP(state_dim, action_dim * 2)
        self.action_limit = action_limit

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action_scaled = action * self.action_limit
        return action_scaled, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_limit


class Critic(nn.Module):
    """Twin Q network for clipped double-Q learning."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_temperature: float = 0.2
    replay_size: int = 300_000


class SACAgent:
    """SAC implementation for continuous actions and learned reward signals."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_limit: float,
        device: str,
        config: SACConfig,
    ) -> None:
        self.device = torch.device(device)
        self.config = config

        self.actor = Actor(state_dim, action_dim, action_limit).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.log_alpha = torch.tensor(np.log(config.init_temperature), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -float(action_dim)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config.replay_size, self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(state_t)
        else:
            action, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch_size: int) -> Dict[str, float]:
        if self.replay_buffer.size < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0, "alpha": float(self.alpha.item())}

        batch = self.replay_buffer.sample(batch_size=batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = rewards + (1.0 - dones) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        sampled_actions, log_prob = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update_targets(self.config.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def _soft_update_targets(self, tau: float) -> None:
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def state_dict(self) -> Dict[str, object]:
        """Serialize SAC model, optimizers, and replay buffer."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().item(),
            "target_entropy": self.target_entropy,
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore SAC model, optimizers, and replay buffer from checkpoint."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])

        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])

        self.log_alpha = torch.tensor(
            float(state["log_alpha"]),
            device=self.device,
            requires_grad=True,
        )
        self.target_entropy = float(state["target_entropy"])

        # Ensure optimizer points to current log_alpha tensor after restore.
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.alpha_opt.load_state_dict(state["alpha_opt"])

        self.replay_buffer.load_state_dict(state["replay_buffer"])
