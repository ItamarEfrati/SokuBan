import os
import pickle
import random
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from tqdm import tqdm

from env.soko_pap import PushAndPullSokobanEnv


class Solver(ABC):

    def __init__(self,
                 log_dir,
                 seed,
                 device,
                 lr: float,
                 gamma: float,
                 num_steps,
                 num_episodes,
                 val_every,
                 log_every_n_epochs,
                 num_evaluations):

        self.num_evaluations = num_evaluations
        self.log_dir = log_dir
        self.games_won = 0
        self.games_results = []
        self.winning_ratio = []
        self.average_evaluation_rewards = []
        self.losses = []
        self.seed = seed
        self.train_env = self.reset()

        self.gamma = gamma
        self.val_every = val_every
        self.lr = lr

        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.device = device

        self.log_every_n_epochs = log_every_n_epochs
        self.current_total_step = 0
        self.current_epoch = 0

        self.optimizer = None

    def reset(self):
        """Resents the environment and updates the state."""
        if self.seed is not None:
            random.seed(self.seed)
        env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
        return env

    def train(self):
        self.set_optimizer()
        # self.eval()
        print("Starting training")
        for i_episode in range(1, self.num_episodes + 1):

            episode_loss = self.run_single_epoch()

            self.update_statistics(episode_loss)
            if self.current_epoch % self.log_every_n_epochs == 0:
                self.log_update()

            self.current_epoch += 1
            self.on_epoch_end()

            if self.current_epoch % self.val_every == 0:
                self.eval()
        self.on_train_end()
        self.save_statistics()

    @abstractmethod
    def save_statistics(self):
        pass

    def eval(self):
        # self.save_model()
        print("running evaluation")
        games_won = 0
        reward = 0
        for i in tqdm(range(self.num_evaluations)):
            states, total_reward = self.play_single_game()

            file_path = os.path.join(self.log_dir, 'videos', f"epoch_{self.current_epoch}.mp4")
            games_won += 1 if total_reward > -50 else 0
            reward += total_reward
            if i == 0:
                self._save_video(states=states, file_path=file_path)

        self.log_evaluation_results(games_won, reward)
        # self.save_statistics()

    @staticmethod
    def _save_video(states, file_path):
        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (256, 256))
        for frame in states:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            out.write(frame)
        out.release()

    def log_evaluation_results(self, games_won, rewards):
        winning_ratio = games_won / self.num_evaluations
        average_reward = rewards / self.num_evaluations
        self.winning_ratio.append(winning_ratio)
        self.average_evaluation_rewards.append(average_reward)
        log = f"epoch {self.current_epoch} " \
              f"step {self.current_total_step} " \
              f"winning ratio {winning_ratio:.3f} " \
              f"average reward {average_reward:.3f} " \
              f"total reward {rewards:.3f}"
        print(log)

    def log_update(self):
        update = f"epoch {self.current_epoch} " \
                 f"step {self.current_total_step} " \

        update = self.add_statistic_to_log(update)

        print(update)

    # region Abstract

    @abstractmethod
    def set_optimizer(self):
        pass

    @abstractmethod
    def optimize_model(self):
        pass

    @abstractmethod
    def play_single_game(self):
        pass

    @abstractmethod
    def play_step(self, env, state, epsilon, is_train=True):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            :param env:
            :param state:
            :param is_train:
            :param epsilon: value to determine likelihood of taking a random action

        Returns:
            reward, done, state

        """
        pass

    @abstractmethod
    def get_action(self, env, state, epsilon):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def get_current_state(self, env):
        pass

    @abstractmethod
    def add_statistic_to_log(self, update):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def save_hparams(self):
        pass

    @abstractmethod
    def run_single_epoch(self):
        pass

    @abstractmethod
    def update_statistics(self, episode_loss):
        pass

    @abstractmethod
    def on_train_end(self):
        pass
    # endregion
