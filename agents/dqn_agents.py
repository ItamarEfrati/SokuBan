from typing import Tuple

import numpy as np
import torch
from torch import nn

from agents.basic_agent import Agent
from data_modules.rl_dataset import Experience
from models.dqn import DQN, D3QN


class DQNAgent(Agent):
    """Basic DQN Model."""

    def __init__(
            self,
            train_env,
            val_env,
            seed,
            basic_net,
            double_q_learning=True,
            batch_size: int = 16,
            lr: float = 1e-3,
            gamma: float = 0.9,
            sync_rate: int = 10,
            replay_size: int = 10_000,
            warm_start_size: int = 6000,
            eps_last_frame: int = 1000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            episode_length: int = 1000,
            val_every=5
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            # warm_start_steps: max episode reward in the environment
        """
        self.save_hyperparameters()
        super().__init__(train_env,
                         val_env,
                         seed,
                         batch_size,
                         lr,
                         gamma,
                         replay_size,
                         warm_start_size,
                         eps_last_frame,
                         eps_start,
                         eps_end,
                         episode_length,
                         val_every)
        self.target_net = basic_net(output_size=train_env.action_space.n)
        self.policy_net = basic_net(output_size=train_env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.populate()

    # region Agent

    def populate(self) -> None:
        for _ in range(self.hparams.warm_start_size):
            _, _, self.state = self.play_step(self.train_env, self.state, self.policy_net, epsilon=1.0)

    @torch.no_grad()
    def play_single_game(self):
        is_done = False
        state = self.reset(self.val_env)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        states = []
        total_reward = 0
        while not is_done:
            reward, is_done, state = self.play_step(self.val_env, state, policy=self.policy_net, is_train=False,
                                                    epsilon=epsilon)
            total_reward += reward
            states.append(state)

        return states, total_reward

    @torch.no_grad()
    def play_step(
            self,
            env,
            state,
            policy: nn.Module,
            is_train=True,
            epsilon: float = 0.0,
            device: str = "cuda",

    ):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            policy: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done, state
        """

        action = self.get_action(env, state, policy, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = env.step(action)

        exp = Experience(state, action, reward, done, new_state)

        if is_train:
            self.buffer.append(exp)

        state = new_state
        if done and is_train:
            state = self.reset(env)
        return reward, done, state

    def get_action(self, env, state, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state = self.dataset.transform(state)

            if device not in ["cpu"]:
                state = state.cuda(self.device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    # endregion

    def training_step(self, batch, nb_batch):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        self.log("epsilon", epsilon, prog_bar=True)

        # step through environment with agent
        reward, done, self.state = self.play_step(env=self.train_env,
                                                  state=self.state,
                                                  policy=self.policy_net,
                                                  epsilon=epsilon,
                                                  device=device)
        self.episode_reward += reward
        self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", torch.tensor(self.total_reward, dtype=torch.float32), prog_bar=True)
        self.log("episode_reward", torch.tensor(self.episode_reward, dtype=torch.float32), prog_bar=True,
                 on_epoch=False, on_step=True)
        self.log("steps", torch.tensor(self.global_step, dtype=torch.float32), logger=False, prog_bar=True)

        return {"loss": loss}

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        q_values = self.policy_net(states).gather(dim=1, index=actions.long().unsqueeze(0)).squeeze()

        with torch.no_grad():
            if self.hparams.double_q_learning:
                actions = actions.long()
                next_q_values = self.target_net(next_states).gather(dim=1, index=actions.long().unsqueeze(0)).squeeze()
            else:
                next_q_values = self.target_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0

        expected_state_action_values = next_q_values * self.hparams.gamma + rewards

        return nn.functional.mse_loss(q_values, expected_state_action_values)
