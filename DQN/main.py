import os
import gc
import torch
import numpy as np
import gymnasium as gym
import pygame
import matplotlib.pyplot as plt
import datetime
# import argparse

from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()

def seed_everything(seed=42):
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class RLModel:
    def __init__(self, hyperparams):
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]

        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]

        self.memory_capacity = hyperparams["memory_capacity"]

        self.num_states = hyperparams["num_states"]
        self.map_size = hyperparams["map_size"]
        self.is_slippery = hyperparams["is_slippery"]
        self.render_fps = hyperparams["render_fps"]

        self.env = gym.make('FrozenLake-v1', map_name=f"{self.map_size}x{self.map_size}",
                            is_slippery=self.is_slippery, max_episode_steps=self.max_steps,
                            render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps

        self.agent = DQNAgent(env=self.env,
                              epsilon_max=self.epsilon_max,
                              epsilon_min=self.epsilon_min,
                              epsilon_decay=self.epsilon_decay,
                              clip_grad_norm=self.clip_grad_norm,
                              learning_rate=self.learning_rate,
                              discount=self.discount_factor,
                              memory_capacity=self.memory_capacity)

    def state_preprocess(self, state, num_states):
        onehot_vector = torch.zeros(num_states, dtype=torch.float32, device=device)
        onehot_vector[state] = 1
        return onehot_vector

    def train(self):
        total_steps = 0
        self.reward_history = []

        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=42)
            state = self.state_preprocess(state, num_states=self.num_states)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                next_state = self.state_preprocess(next_state, num_states=self.num_states)

                self.agent.replay_memory.store(state, action, next_state, reward, done)

                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state = next_state
                episode_reward += reward
                step_size += 1

            self.reward_history.append(episode_reward)
            total_steps += step_size

            self.agent.update_epsilon()

            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon_max:.2f}")
            print(result)
        self.plot_training(episode)

    def test(self, max_episodes):
        state_dict = torch.load(self.RL_load_path)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace("FC", "layers")
            new_state_dict[new_key] = state_dict[key]
        self.agent.main_network.load_state_dict(new_state_dict)
        self.agent.main_network.eval()

        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=42)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                state = self.state_preprocess(state, num_states=self.num_states)
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                state = next_state
                episode_reward += reward
                step_size += 1

            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)

        pygame.quit()

    def plot_training(self, episode):
        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        if episode == self.max_episodes:
            plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        if episode == self.max_episodes:
            plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    '''
    # Argument parser --> canceled and changed to yaml file 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, required=True, help='Train or Evaluate')
    parser.add_argument('--RL_load_path', type=str, required=True, help='Path to model')
    parser.add_argument('--map_size', type=int, default=8)
    args = parser.parse_args()
    model = RLModel(train_mode=args.train_mode,
                    RL_load_path=args.RL_load_path
                    map_size=args.map_size)
    '''


    seed_everything(42)
    train_mode = False  # True
    render = not train_mode
    map_size = 4  # 4 or 8
    RL_hyperparams = {
        "train_mode": train_mode,
        #"RL_load_path": f'./DQN/{map_size}x{map_size}_map/final_weights' + '_' + '3000' + '.pth',

        # map_size == 8
        #"RL_load_path": f'./DQN/{map_size}x{map_size}_map/final_weights' + '2024-07-06 15:38:48.494434_' + '3000' + '.pth',
        # map_size == 4
        "RL_load_path": f'./DQN/{map_size}x{map_size}_map/final_weights' + '2024-07-06 16:56:40.697539_' + '3000' + '.pth',

        "save_path": f'./DQN/{map_size}x{map_size}_map/final_weights{datetime.datetime.now()}',
        "save_interval": 500,

        "clip_grad_norm": 3,
        "learning_rate": 6e-4,
        "discount_factor": 0.93,
        "batch_size": 32,
        "update_frequency": 10,
        "max_episodes": 3000 if train_mode else 5,
        "max_steps": 200,
        "render": render,

        "epsilon_max": 0.999 if train_mode else -1,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.999,

        "memory_capacity": 4000 if train_mode else 0,

        "map_size": map_size,
        "is_slippery": False,  # True
        "num_states": map_size ** 2,
        "render_fps": 6,
    }

    model = RLModel(RL_hyperparams)
    if train_mode:
        model.train()
    else:
        model.test(max_episodes=RL_hyperparams['max_episodes'])
