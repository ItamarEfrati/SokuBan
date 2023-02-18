import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple
from torch import nn

from agents.basic_agent import Agent
from models.actor_critic import ActorCritic


class ActorCriticAgent(Agent):
    def __init__(self, train_env, val_env, seed):
        super().__init__(train_env, val_env, seed)

        self.actor_critic = ActorCritic(train_env.action_space.n)

    def get_action(self, env, state, net: nn.Module, epsilon: float, device: str) -> int:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = self.actor_critic(state)
            _, action = torch.max(q_values, dim=1)

        return action

    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)

        probs = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()

        return logit, action_log_probs, value, entropy

    def play_single_game(self):
        pass

    def play_step(self, env, state, is_train=True, epsilon: float = 0.0, device: str = "cuda") -> Tuple[
        float, bool, float]:
        pass

    def populate(self) -> None:
        pass
