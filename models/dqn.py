import pickle

import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn, Tensor
from typing import Tuple, OrderedDict
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from data_modules.rl_dataset import ReplayBuffer, Experience, RLDataset


class DQN(nn.Module):
    def __init__(self,
                 output_size,
                 kernel_size_1=(9, 9),
                 kernel_size_2=(7, 7),
                 kernel_size_3=(5, 5),
                 kernel_size_4=(3, 3),
                 kernel_size_5=(3, 3),
                 stride_1=(1, 1),
                 stride_2=(1, 1),
                 stride_3=(1, 1),
                 stride_4=(1, 1),
                 stride_5=(1, 1),
                 out_channels_1=16,
                 out_channels_2=8,
                 out_channels_3=4,
                 out_channels_4=4,
                 out_channels_5=1,
                 hidden_size_1=1024,
                 hidden_size_2=128,
                 ):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels_1, kernel_size=kernel_size_1, stride=stride_1)
        self.pool1 = nn.MaxPool2d(kernel_size_1, stride=stride_1)

        self.conv2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size_2,
                               stride=stride_2)
        self.pool2 = nn.MaxPool2d(kernel_size_2, stride=stride_2)

        self.conv3 = nn.Conv2d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=kernel_size_3,
                               stride=stride_3)
        self.pool3 = nn.MaxPool2d(kernel_size_3, stride=stride_3)

        self.conv4 = nn.Conv2d(in_channels=out_channels_3, out_channels=out_channels_4, kernel_size=kernel_size_4,
                               stride=stride_4)
        self.pool4 = nn.MaxPool2d(kernel_size_4, stride=stride_4)

        self.conv5 = nn.Conv2d(in_channels=out_channels_4, out_channels=out_channels_5, kernel_size=kernel_size_5,
                               stride=stride_5)
        self.pool5 = nn.MaxPool2d(kernel_size_5, stride=stride_5)

        in_features = 4624
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_size_1)
        self.fc2 = nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2)
        self.fc3 = nn.Linear(in_features=hidden_size_2, out_features=output_size)

    def forward(self, x):
        x_in = x
        try:
            x = F.relu(self.pool1(self.conv1(x)))
            x = F.relu(self.pool2(self.conv2(x)))
            x = F.relu(self.pool3(self.conv3(x)))
            x = F.relu(self.pool4(self.conv4(x)))
            x = F.relu(self.pool5(self.conv5(x)))
            x = F.relu(self.fc1(x.flatten(1)))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        except:
            print(1)
        return x


class DQNAgent(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            train_env,
            val_env,
            seed,
            batch_size: int = 64,
            lr: float = 1e-3,
            gamma: float = 0.99,
            sync_rate: int = 10,
            replay_size: int = 1500,
            warm_start_size: int = 1000,
            eps_last_frame: int = 250,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            episode_length: int = 500,
            val_every=5
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
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
        super().__init__()
        self.save_hyperparameters()
        self.train_env = train_env
        self.val_env = val_env
        self.seed = seed
        self.state = None
        self.state = self.reset(self.train_env)
        self.reset(self.val_env)
        obs_size = self.train_env.observation_space.shape[0]
        n_actions = self.train_env.action_space.n

        self.target_net = DQN(output_size=n_actions)
        self.policy_net = DQN(output_size=n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_size)
        self.dataset = RLDataset(self.buffer, sample_size=self.hparams.episode_length)

    # region Agent

    def reset(self, env):
        """Resents the environment and updates the state."""
        if self.seed is not None:
            pl.seed_everything(self.seed)
        return env.reset()

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

    @torch.no_grad()
    def play_single_game(self):
        is_done = False
        state = self.reset(self.val_env)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        states = []
        total_reward = 0
        while not is_done:
            reward, is_done, state = self.play_step(self.val_env, state, net=self.policy_net, is_train=False, epsilon=epsilon)
            total_reward += reward
            states.append(state)

        return states, total_reward

    @torch.no_grad()
    def play_step(
            self,
            env,
            state,
            net: nn.Module,
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

        action = self.get_action(env, state, net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = env.step(action)

        exp = Experience(state, action, reward, done, new_state)

        if is_train:
            self.buffer.append(exp)

        state = new_state
        if done and is_train:
            state = self.reset(env)
        return reward, done, state

    # endregion

    # region move

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            _, _, self.state = self.play_step(self.train_env, self.state, self.policy_net, epsilon=1.0)
        with open('experiences2', 'rb') as f:
            exps = pickle.load(f)
        for i in range(5):
            for exp in exps:
                self.buffer.append(exp)

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.policy_net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    # endregion

    # region Lightning

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
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
                                                  net=self.policy_net,
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
        self.log("steps", torch.tensor(self.global_step, dtype=torch.float32), logger=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.hparams.val_every == 0:
            states, total_reward = self.play_single_game()
            video = torch.tensor(np.stack(states).transpose(0, 3, 1, 2)).unsqueeze(0)
            self.logger.experiment.add_video(f'single game', video, self.current_epoch)


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

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    # endregion
