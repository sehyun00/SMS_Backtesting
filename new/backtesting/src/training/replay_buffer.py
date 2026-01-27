import random
import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    Experience Replay Buffer (FIFO with Capacity Limit)
    Stores transitions (state, action, reward, next_state, done)
    for off-policy RL training (DDPG/Hybrid).
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience.
        Oldest experience is automatically removed if full.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly sample a batch of experiences.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            list(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            list(next_states),
            np.array(dones).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)
