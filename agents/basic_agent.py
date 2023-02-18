import pickle
from abc import ABC, abstractmethod
from typing import Tuple, OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_modules.rl_dataset import ReplayBuffer, RLDataset, Experience


class Agent(LightningModule, ABC):
    """Basic DQN Model."""

    def __init__(
            self,
            train_env,
            val_env,
            seed,
            batch_size: int = 16,
            lr: float = 1e-3,
            gamma: float = 0.9,
            replay_size: int = 10_000,
            warm_start_size: int = 8000,
            eps_last_frame: int = 2000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            episode_length: int = 1000,
            val_every=5
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            train_env: gym environment for training
            val_env: gym environment for evaluation
            gamma: discount factor
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            # warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.train_env = train_env
        self.val_env = val_env
        self.seed = seed
        self.state = None
        self.state = self.reset(self.train_env)
        self.reset(self.val_env)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.total_reward = 0
        self.episode_reward = 0
        self.dataset = RLDataset(self.buffer, sample_size=self.hparams.episode_length)

    # region Agent

    def reset(self, env):
        """Resents the environment and updates the state."""
        if self.seed is not None:
            pl.seed_everything(self.seed)
        return env.reset()

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
            is_train=True,
            epsilon: float = 0.0,
            device: str = "cuda",

    ) -> Tuple[float, bool, float]:
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
    def populate(self) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        pass

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    # endregion

    # region Lightning

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizer = Adam(self.policy_net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=1
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.hparams.val_every == 0:
            states, total_reward = self.play_single_game()
            video = torch.tensor(np.stack(states).transpose(0, 3, 1, 2)).unsqueeze(0)
            self.logger.experiment.add_video(f'single game', video, self.global_step)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    # endregion
