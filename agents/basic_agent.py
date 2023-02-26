import pickle
import random
from abc import ABC, abstractmethod
from typing import Tuple, OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.utils.data import DataLoader

from env.soko_pap import PushAndPullSokobanEnv


class Agent(LightningModule, ABC):
    """Basic DQN Model."""

    def __init__(
            self,
            seed,
            lr: float = 1e-3,
            gamma: float = 0.9,
            eps_last_frame: int = 2000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            val_every=5
    ) -> None:
        """
        Args:
            lr: learning rate
            gamma: discount factor
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            # warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.seed = seed
        self.state = None
        self.train_env = self.reset()
        self.state = self.train_env.render(mode='rgb_array')
        self.total_reward = -50
        self.episode_reward = 0

    # region Agent

    def reset(self):
        """Resents the environment and updates the state."""
        if self.seed is not None:
            # pl.seed_everything(self.seed)
            random.seed(self.seed)
        env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
        return env

    @abstractmethod
    def get_action(self, env, state, net: nn.Module, epsilon: float, device: str) -> int:
        pass

    @abstractmethod
    @torch.no_grad()
    def play_single_game(self):
        pass

    @abstractmethod
    @torch.no_grad()
    def play_step(
            self,
            env,
            state,
            policy: nn.Module,
            is_train=True,
            epsilon: float = 0.0,
            device: str = "cuda",

    ) -> Tuple[float, bool, float, torch.tensor]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done, state
        """
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    # endregion

    # region Lightning

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch % self.hparams.val_every == 0:
            states, total_reward, q_values_t = self.play_single_game()
            video = torch.tensor(np.stack(states).transpose(0, 3, 1, 2)).unsqueeze(0)
            self.logger.experiment.add_video(f'single game', video, self.global_step)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    # endregion
