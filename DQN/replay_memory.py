import torch
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.state_buffer = deque(maxlen=capacity)
        self.action_buffer = deque(maxlen=capacity)
        self.next_state_buffer = deque(maxlen=capacity)
        self.reward_buffer = deque(maxlen=capacity)
        self.done_buffer = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.next_state_buffer.append(next_state)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    def sample(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(self.state_buffer[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        actions = torch.as_tensor([self.action_buffer[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack([torch.as_tensor(self.next_state_buffer[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.reward_buffer[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.done_buffer[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.done_buffer)
