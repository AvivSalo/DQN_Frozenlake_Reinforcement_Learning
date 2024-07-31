import torch
import numpy as np
import torch.nn as nn

from dqn_network import DeepQNetwork
from replay_memory import ExperienceReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, clip_grad_norm, learning_rate, discount, memory_capacity):
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.replay_memory = ExperienceReplayBuffer(memory_capacity)

        self.main_network = DeepQNetwork(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device)
        self.target_network = DeepQNetwork(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.SGD(self.main_network.parameters(), lr=learning_rate)



    def select_action(self, state):
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()
        with torch.no_grad():
            Q_values = self.main_network(state)
            return torch.argmax(Q_values).item()

    def learn(self, batch_size, done):
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        predicted_q = self.main_network(states).gather(dim=1, index=actions)

        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0]
        next_target_q_value[dones] = 0
        y_js = rewards + (self.discount * next_target_q_value)
        loss = self.criterion(predicted_q, y_js)

        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss)
            self.running_loss = 0
            self.learned_counts = 0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        torch.save(self.main_network.state_dict(), path)
