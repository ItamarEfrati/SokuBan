import torch
import torchvision.transforms as T

from collections import namedtuple, deque
from collections.abc import Iterator
from typing import Tuple

import numpy as np
from torch.utils.data import IterableDataset

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", 'reshaped_reward', "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity, device) -> None:
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.CenterCrop(80),

        ])

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, reshaped_rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            torch.concat(states).unsqueeze(1),
            torch.tensor(actions, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(reshaped_rewards, dtype=torch.float32, device=self.device),
            np.array(dones, dtype=bool),
            torch.concat(next_states).unsqueeze(1),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200, is_sample=True) -> None:
        self.is_sample = is_sample
        self.buffer = buffer
        self.sample_size = sample_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.CenterCrop(80)
        ])

    def __iter__(self) -> Iterator[Tuple]:
        if self.is_sample:
            states, actions, rewards, reshape_rewards, dones, new_states = self.buffer.sample(self.sample_size)
        else:
            states, actions, rewards, reshape_rewards, dones, new_states = self.buffer.get()
        for i in range(len(dones)):
            next_state = new_states[i] if new_states[i] is None else self.transform(new_states[i])
            yield self.transform(states[i]), actions[i], rewards[i], reshape_rewards[i], dones[i], next_state

    def __len__(self):
        return self.sample_size
