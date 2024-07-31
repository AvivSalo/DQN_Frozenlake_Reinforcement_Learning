# ❄️ DQN Frozenlake Reinforcement Learning ❄️
This repository contains an implementation of a Deep Q-Network (DQN) using a Reinforcement Learning (RL) agent in the Frozen Lake environment from GYM. The aim is to train the RL agent to navigate the frozen lake and reach the goal without falling into holes.

## Features

- **DQN Implementation:** A deep learning model is implemented to approximate the Q-value function, enabling the agent to make optimal decisions.
- **GYM Environment:** Utilizes OpenAI's GYM toolkit to create and manage the Frozen Lake environment, providing a robust and flexible simulation platform.
- **Training and Evaluation:** The agent is trained using the DQN algorithm, and performance is evaluated based on the agent's ability to navigate the lake successfully.
- **Customizable Parameters:** Various parameters such as learning rate, discount factor, and exploration rate can be adjusted to experiment with different training configurations.
- **Visualization:** Includes tools to visualize the agent's progress and performance metrics over time.

## Requirements

- Python 3.x
- OpenAI GYM - gymnasium
- PyTorch
- NumPy
- Matplotlib
- PyGame

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AvivSalo/DQN_Frozenlake_Reinforcement_Learning.git
    cd DQN_Frozenlake_Reinforcement_Learning
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Train the agent:
   choose your map size 4x4 or 8x8 by the parameter map_size and set train_mode to True
    ```bash
    python main.py
    ```
2. Evaluate the agent:
   set train_mode to False with the corresponding map_size and path to model
    ```bash
    python main.py
    ```

## Repository Structure
DQN/

    ├── main.py
    ├── dqn_agent.py
    ├── dqn_network.py
    └── replay_memory.py
