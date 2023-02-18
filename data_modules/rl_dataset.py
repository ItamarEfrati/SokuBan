import torchvision.transforms as T

from collections import namedtuple, deque
from collections.abc import Iterator
from typing import Tuple

import numpy as np
from torch.utils.data import IterableDataset

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            next_states,
        )

    # def sample(self, batch_size: int) -> Tuple:
    #     indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    #     states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
    #
    #     return (
    #         torch.tensor(states),
    #         torch.tensor(actions),
    #         torch.tensor(rewards, dtype=torch.float32),
    #         torch.tensor(dones, dtype=torch.bool),
    #         torch.tensor(next_states),
    #     )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.CenterCrop(80)
        ])

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            next_state = new_states[i] if new_states[i] is None else self.transform(new_states[i])
            yield self.transform(states[i]), actions[i], rewards[i], dones[i], next_state

    def __len__(self):
        return self.sample_size
