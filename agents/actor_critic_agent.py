import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple
from torch import nn
from torch.optim import Adam

from agents.basic_agent import Agent
from data_modules.rl_dataset import Experience, ReplayBuffer, RLDataset
from models.actor_critic import ActorCritic


class ActorCriticAgent(Agent):

    def __init__(self,
                 train_env,
                 val_env,
                 seed,
                 batch_size=16,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 eps_last_frame: int = 32_000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.01,
                 val_every=5
                 ):
        self.save_hyperparameters()
        super().__init__(train_env,
                         val_env,
                         seed,
                         lr,
                         gamma,
                         eps_last_frame,
                         eps_start,
                         eps_end,
                         val_every
                         )
        self.actor_critic_policy = ActorCritic(train_env.action_space.n)

        self.buffer = ReplayBuffer(500)
        self.dataset = RLDataset(self.buffer, sample_size=0, is_sample=False)
        self.play_single_game_train()

    # region Agent

    def play_single_game_train(self):
        is_done = False
        state = self.reset(self.train_env)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        while not is_done:
            reward, is_done, state = self.play_step(self.train_env, state, policy=self.actor_critic_policy,
                                                    is_train=True, epsilon=epsilon)

    def get_action(self, env, state, net: nn.Module, epsilon: float, device: str) -> int:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state = self.dataset.transform(state)

            if device not in ["cpu"]:
                state = state.cuda(self.device)

            logits, _ = net(state)
            probs = F.softmax(logits, dim=1)

            action = probs.max(1)[1]
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
            reward, is_done, state = self.play_step(self.val_env, state, policy=self.actor_critic_policy,
                                                    is_train=False, epsilon=epsilon)
            total_reward += reward
            states.append(state)

        return states, total_reward

    def play_step(self, env, state, policy: nn.Module, is_train=True, epsilon: float = 0.0, device: str = "cuda") -> \
            Tuple[float, bool, float]:
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

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        logits, states_value = self.actor_critic_target(states)
        _, next_states_value = self.actor_critic_target(next_states)
        next_states_value[dones] = 0.0
        expected_state_values = next_states_value * self.hparams.gamma + rewards
        advantage = expected_state_values - states_value
        log_probs = F.log_softmax(logits)
        log_probs = log_probs.gather(1, actions.long().unsqueeze(0))

        value_loss = torch.pow(advantage, 2).mean(-1)
        policy_loss = (advantage * log_probs).mean(-1)

        loss = value_loss + policy_loss
        return loss

    # endregion

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizer = Adam(self.actor_critic_policy.parameters(), lr=self.hparams.lr)
        return optimizer

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
                                                  policy=self.actor_critic_policy,
                                                  epsilon=epsilon,
                                                  device=device)
        self.episode_reward += reward
        self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = self.calculate_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        # if self.global_step % self.hparams.sync_rate == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", torch.tensor(self.total_reward, dtype=torch.float32), prog_bar=True)
        self.log("episode_reward", torch.tensor(self.episode_reward, dtype=torch.float32), prog_bar=True,
                 on_epoch=True, on_step=True)
        self.log("steps", torch.tensor(self.global_step, dtype=torch.float32), logger=False, prog_bar=True)

        self.play_single_game_train()
        return {"loss": loss}
