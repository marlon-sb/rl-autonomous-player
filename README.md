# Reinforcement Learning for Text-Based Games

## Table of Contents
- [Project Overview](#project-overview)
- [Algorithms & Models](#algorithms--models)
  - [1. Tabular Q-Learning](#1-tabular-q-learning)
  - [2. Q-Learning with Linear Approximation](#2-q-learning-with-linear-approximation)
  - [3. Deep Q-Network (DQN)](#3-deep-q-network-dqn)
- [Environment & State](#environment--state)
  - [The "Home World"](#the-home-world)
  - [State Representation](#state-representation)
- [Installation](#installation)
- [Usage](#usage)
- [Final Results](#final-results)

***

## Project Overview

This project implements a series of reinforcement learning agents to learn control policies for a text-based game. This project is from the MIT course **"Machine Learning with Python: From Linear Models to Deep Learning"**.

The goal is to train an autonomous agent to complete quests within a virtual "Home World" where all interactions are through text. The agent must learn to map textual descriptions of its state (e.g., "You are in the living room" and "You are hungry") to optimal commands (e.g., "go east") to maximize its cumulative reward.

The project explores three levels of agent complexity:
1.  **Tabular Q-Learning**: A baseline using a simple lookup table.
2.  **Q-Learning with Linear Approximation**: A more scalable approach using feature vectors.
3.  **Deep Q-Network (DQN)**: An advanced agent using a neural network to approximate Q-values.

***

## Algorithms & Models

### 1. Tabular Q-Learning

Found in `agent_tabular.py`, this agent is the first implementation and serves as a baseline. It treats every unique state description (room + quest) as a discrete, unique index.

-   **Q-Table**: A large 4D NumPy array `Q[room_idx, quest_idx, action, object]` stores the learned value for every possible state-action pair.
-   **Update Rule**: Implements the classic Bellman update rule:
    $Q(s,c) \leftarrow (1-\alpha)Q(s,c) + \alpha [R(s,c) + \gamma \max_{c'} Q(s', c')]$

### 2. Q-Learning with Linear Approximation

Found in `agent_linear.py`, this agent addresses the scalability problem of the tabular method. Instead of storing a value for every state, it approximates the Q-function with a linear model:
$$Q(s, c) \approx \phi(s, c)^T \theta$$

-   **State Features**: The textual state is converted into a **Bag-of-Words (BoW)** feature vector $\psi_R(s)$.
-   **Q-Function**: The agent learns a separate weight vector $\theta_c$ for each action $c$. The Q-value is the dot product of the state vector and the action's corresponding weight vector: $Q(s, c) \approx \psi_R(s)^T \theta_c$.
-   **Update Rule**: Implements Stochastic Gradient Descent (SGD) to update the `theta` matrix based on the TD error.

### 3. Deep Q-Network (DQN)

Found in `agent_dqn.py`, this is the most advanced agent. It replaces the linear model with a PyTorch neural network to approximate the Q-function.

-   **Network Architecture**: The `DQN` class uses a feed-forward network with one hidden layer. It has a single "body" that encodes the state vector and **two "heads"** (two separate output layers) to predict Q-values for actions and objects independently.
-   **Loss Function**: The network is trained by minimizing the **Mean Squared Error (MSE)** between the predicted Q-value and the TD target.
-   **Update Rule**: The network's weights are updated using backpropagation and the `SGD` optimizer.

***

## Environment & State

### The "Home World"
The agent interacts with a small text-based game environment consisting of four rooms with connecting pathways:
-   Living Room (contains a TV)
-   Garden (contains a bike)
-   Bedroom (contains a bed)
-   Kitchen (contains an apple)

At the start of each episode, the agent is given a quest (e.g., "You are hungry"), and its goal is to navigate to the correct room and interact with the correct object to get a positive reward.

### State Representation
The true state (room, quest) is **hidden**. The agent only receives text descriptions.
-   **Tabular**: A dictionary maps each unique `(room_desc, quest_desc)` string pair to a unique integer index.
-   **Linear & DQN**: A **Bag-of-Words (BoW)** model is used to convert the combined `room_desc + quest_desc` string into a high-dimensional feature vector.

***

## Installation

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone [https://github.com/marlon-sb/rl-autonomous-player.git](https://github.com/marlon-sb/rl-autonomous-player.git)
cd rl-autonomous-player
```
### Step 2: Create and Activate a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate (on macOS/Linux)
source venv/bin/activate

# Activate (on Windows)
.\venv\Scripts\activate.ps1
```
### Step 3: Install Dependencies
Install the required Python libraries (**NumPy**, **Matplotlib**, **PyTorch**, etc.):

```bash
pip install -r requirements.txt
```
## Usage
The project is divided into three main agent files. You can run each experiment independently.

```bash
# Run the Tabular Q-Learning agent
python agent_tabular.py

# Run the Q-Learning with Linear Approximation agent
python agent_linear.py

# Run the Deep Q-Network (DQN) agent
python agent_dqn.py
```
Each script will run the full experiment (multiple runs and epochs) and automatically generate a Matplotlib plot showing the average reward over time.

## Final Results

The performance of each agent was measured by its **average episodic reward** during testing, once the model converged.

| Agent | Convergence Reward (Avg) |
|--------|---------------------------|
| Q-Learning (Linear Approximation) | ~0.42 |
| Deep Q-Network (DQN) | ~0.50 |

As expected, the **DQN agent** achieved the highest performance, learning a more optimal policy than the linear model.

---

### Q-Learning with Linear Approximation (Plot)
The linear model converged to an average reward of approximately **0.42**.  
The high variance suggests the model struggles to consistently find the optimal policy.

---

### Deep Q-Network (DQN) (Plot)
The DQN agent converged to a higher average reward of approximately **0.50**.  
The plot also shows **faster and more stable convergence**, demonstrating its superior ability to model complex state–action values.
