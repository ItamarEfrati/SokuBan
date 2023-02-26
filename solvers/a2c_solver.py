import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch import nn
from torch.optim import Adam

from data_modules.rollout import RolloutStorage
from models.actor_critic import ActorCritic
from solvers.basic_solver import Solver
from utils.multiprocess_env import SubprocVecEnv
from utils.reward_reshape import get_potential


class A2CSolver(Solver):
    def __init__(self,
                 log_dir,
                 seed,
                 device,
                 lr: float = 7e-4,
                 gamma: float = 0.99,
                 num_steps=10,
                 num_episodes=100_000,
                 val_every=1000,
                 log_every_n_epochs=50,
                 num_envs=2,
                 num_evaluations=20,
                 is_reward_shaping=True
                 ):
        super().__init__(log_dir, seed, device, lr, gamma, num_steps, num_episodes,
                         val_every, log_every_n_epochs, num_evaluations)

        self.state_shape = (1, 80, 80)
        self.entropy = []
        self.value_loss = []
        self.action_loss = []
        self.is_reward_shaping = is_reward_shaping

        self.entropy_coef = 0.1
        self.value_loss_coef = 1
        self.num_envs = num_envs
        self.policy_net = ActorCritic(num_actions=self.train_env.action_space.n).to(self.device)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.CenterCrop(80),

        ])
        print("Initializing envs")
        env_list = [self.make_env() for _ in range(num_envs)]
        self.train_env = SubprocVecEnv(env_list)
        print("Initializing Rollout")
        self.rollout = RolloutStorage(num_steps=num_steps, num_envs=num_envs, state_shape=self.state_shape,
                                      device=device)
        self.save_hparams()

    def make_env(self):
        def _thunk():
            env = self.reset()
            return env

        return _thunk

    def run_single_epoch(self):
        state = self.train_env.render()
        state = self._preprocess_state(state)
        self.rollout.states[0].copy_(state)
        for step in range(self.num_steps):
            reward, done, new_state, action, masks, reshaped_reward = self.play_step(self.train_env, state,
                                                                                     is_train=True)
            self.current_total_step += 1
            new_state = self._preprocess_state(new_state)
            insert_reward = reshaped_reward if self.is_reward_shaping else reward
            self.rollout.insert(step, new_state, action, insert_reward, masks)

        loss, value_loss, action_loss, entropy = self.optimize_model()
        return loss, value_loss, action_loss, entropy

    def set_optimizer(self):
        self.optimizer = Adam(params=self.policy_net.parameters(), lr=self.lr)

    def optimize_model(self):
        critic_loss, actor_loss, entropy = self.calculate_loss(None)
        self.optimizer.zero_grad()
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        self.rollout.after_update()

        return loss, critic_loss, actor_loss, entropy

    def play_single_game(self):
        self.policy_net.eval()
        is_done = False
        val_env = self.reset()
        state = self.get_current_state(val_env)
        states = [state]
        total_reward = 0
        while not is_done:
            reward, is_done, new_state, _, _, _ = self.play_step(val_env,
                                                                 self._preprocess_state(state),
                                                                 is_train=False)
            total_reward += reward.item()
            states.append(new_state)
        self.policy_net.train()
        return states, total_reward

    def play_step(self, env, state, epsilon=None, is_train=True):
        current_potential = get_potential(env)
        action = self.get_action(env, state, is_train=is_train)
        if is_train:
            action_commit = action.detach().cpu().squeeze().numpy()
        else:
            action_commit = action.item() if isinstance(action, torch.Tensor) else action
        new_state, reward, done, _ = env.step(action_commit)
        future_potential = get_potential(env)
        reshaped_reward = reward - future_potential + current_potential
        masks = torch.tensor(1 - done, device=self.device).reshape(-1, 1)
        reward = torch.tensor(reward, device=self.device).reshape(-1, 1)
        reshaped_reward = torch.tensor(reshaped_reward, device=self.device).reshape(-1, 1)
        return reward, done, new_state, action, masks, reshaped_reward

    def get_action(self, env, state, epsilon=None, is_train=True):
        if is_train:
            logit, value = self.policy_net(state)
        else:
            with torch.no_grad():
                logit, value = self.policy_net(state)

        probs = F.softmax(logit, dim=1)
        action = probs.multinomial(num_samples=1)

        return action

    def calculate_loss(self, batch):
        """
        The entropy term in the actor-critic loss function is used to encourage exploration in reinforcement learning
        algorithms. In reinforcement learning, an agent interacts with an environment and learns to take actions that
        maximize a reward signal over time. However, there is a trade-off between exploiting what the agent has
        already learned and exploring new actions that may lead to higher rewards.
        The entropy term in the actor-critic loss function encourages the agent to explore by adding a penalty to the
        loss proportional to the entropy of the policy distribution. The entropy term is calculated based on the policy
        distribution's probability of taking each action. By penalizing low entropy, the agent is encouraged to take
        actions that are not highly predictable and explore more options.
        In other words, the entropy term acts as a regularizer in the training process, promoting diversity in the
        agent's behavior and helping to avoid getting stuck in local optima. By encouraging exploration, the entropy
        term helps the agent discover better policies that may have been missed if it was only focused on exploiting
        what it already knew.
        :param batch:
        :return:
        """
        _, next_value = self.policy_net(self.rollout.states[-1])
        returns = self.rollout.compute_returns(next_value, self.gamma)
        states = self.rollout.states[:-1].view(-1, *self.state_shape)
        actions = self.rollout.actions.view(-1, 1)
        _, action_log_probs, values, entropy = self.policy_net.evaluate_actions(states, actions)
        values = values.view(self.num_steps, self.num_envs, 1)
        action_log_probs = action_log_probs.view(self.num_steps, self.num_envs, 1)

        advantages = returns - values

        critic_loss = F.huber_loss(values, returns)
        actor_loss = -(advantages * action_log_probs).sum()

        return critic_loss, actor_loss, entropy

    def on_epoch_end(self):
        pass

    def get_current_state(self, env):
        state = env.render(mode='rgb_array')
        return state

    def add_statistic_to_log(self, update):
        update += f"loss {self.losses[-1]:.4f} " \
                  f"value loss {self.value_loss[-1]:.4f} " \
                  f"action loss {self.action_loss[-1]:.4f} " \
                  f"entropy {self.entropy[-1]:.4f}"
        return update

    def save_model(self):
        file_path = os.path.join(self.log_dir, 'checkpoints', f"epoch_{self.current_epoch}.pt")
        torch.save({
            'epoch': self.current_epoch,
            'step': self.current_total_step,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'action_loss': self.action_loss,
            'value_loss': self.value_loss,
            'entropy': self.entropy,
            'winning_ratio': self.winning_ratio,
            'average_evaluation_rewards': self.average_evaluation_rewards
        }, file_path)

    def load_model(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_total_step = checkpoint['step']
        self.losses = checkpoint['losses']
        self.action_loss = checkpoint['action_loss']
        self.value_loss = checkpoint['value_loss']
        self.entropy = checkpoint['entropy']
        self.winning_ratio = checkpoint['winning_ratio']
        self.average_evaluation_rewards = checkpoint['average_evaluation_rewards']

    def save_hparams(self):
        d = {
            'log_dir': self.log_dir,
            'seed': self.seed,
            'lr': self.lr,
            'gamma': self.gamma,
            'num_steps': self.num_steps,
            'num_episodes': self.num_episodes,
            'val_every': self.val_every,
            'log_every_n_epochs': self.log_every_n_epochs,
            "num_envs": self.num_envs,
            "is reward shaping": self.is_reward_shaping
        }

        with open(os.path.join(self.log_dir, 'hparams.json'), 'w') as f:
            json.dump(d, f)

    def update_statistics(self, episode_loss):
        episode_loss, value_loss, action_loss, entropy = episode_loss

        self.losses.append(episode_loss.item())
        self.value_loss.append(value_loss.item())
        self.action_loss.append(action_loss.item())
        self.entropy.append(entropy.item())

    def _preprocess_state(self, state):
        if len(state.shape) == 4:
            s_l = []
            for s in state:
                s = self.transform(s).to(self.device)
                s_l.append(s)
            state = torch.stack(s_l)
        else:
            state = self.transform(state).to(self.device).unsqueeze(1)
        return state

    def on_train_end(self):
        self.train_env.close()

    def save_statistics(self):
        with open(os.path.join(self.log_dir, 'statistics.pkl'), 'wb') as f:
            pickle.dump(
                [self.winning_ratio,
                 self.average_evaluation_rewards,
                 self.entropy,
                 self.value_loss,
                 self.action_loss,
                 self.losses],
                f)
