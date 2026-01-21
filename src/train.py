import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import deque, defaultdict
import os

# Import your environment
from src.env import TrafficEnv

# Import your configuration
from src.config import (
    DEFAULT_CONFIG,
    ALPHA,
    GAMMA,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    NUM_EPISODES,
    MODELS_DIR,
    LOGS_DIR,
    MAX_STEPS_PER_EPISODE,
)

# Optional: Try importing DirectML for AMD GPU
try:
    import torch_directml

    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False


# ==========================================
# 1. Q-Learning Agent (Tabular)
# ==========================================
class QLearningAgent:
    def __init__(
        self,
        action_space_size,
        alpha=ALPHA,  # From config
        gamma=GAMMA,  # From config
        epsilon=EPSILON_START,  # From config
        epsilon_decay=EPSILON_DECAY,  # From config
        min_epsilon=EPSILON_MIN,  # From config
    ):
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# ==========================================
# 2. DQN Agent (Neural Network)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        input_dim,
        action_space_size,
        lr=0.0001,
        gamma=GAMMA,  # From config
        epsilon=EPSILON_START,  # From config
        epsilon_decay=EPSILON_DECAY,  # From config
        min_epsilon=EPSILON_MIN,  # From config
    ):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = 64
        self.memory = deque(maxlen=50000)

        self.policy_net = DQN(input_dim, action_space_size)
        self.target_net = DQN(input_dim, action_space_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss(delta=1.0)

        self.step_counter = 0  # Add a counter

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)

        with torch.no_grad():
            # Convert state to tensor on correct device
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        self.step_counter += 1
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Move batch data to device
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.step_counter % 1000 == 0:  # Update every 1000 STEPS, not episodes
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        path = os.path.join(MODELS_DIR, filename)
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")


# ==========================================
# 3. Training Loop
# ==========================================
def train(mode="dqn"):

    # Use NUM_EPISODES from config
    episodes = NUM_EPISODES

    # 1. Configure Environment
    if mode == "q_learning":
        obs_mode = "bucketed"
    else:
        obs_mode = "log"

    # Initialize Env
    env = TrafficEnv(
        config_name="BIDIRECTIONAL",
        obs_mode=obs_mode,
        include_hour=False,
        include_queue=True,
        include_intent=False,
    )

    # FIX: Dynamically check input size to avoid shape errors
    dummy_obs, _ = env.reset()
    obs_dim = dummy_obs.shape[0]
    print(f"Detected Observation Dimension: {obs_dim}")

    # 2. Initialize Agent
    if mode == "q_learning":
        agent = QLearningAgent(action_space_size=env.action_space.n)
    else:
        agent = DQNAgent(input_dim=obs_dim, action_space_size=env.action_space.n)

    print(f"--- Starting Training: {mode.upper()} for {episodes} episodes ---")

    FRAME_SKIP = 10

    for episode in range(episodes):
        obs, _ = env.reset()
        state = obs
        total_reward = 0
        done = False

        while not done:
            # 1. AI makes a decision
            action = agent.get_action(state)

            # 2. Execute that decision for FRAME_SKIP physics steps
            accumulated_reward = 0
            skipped_transitions = 0

            for _ in range(FRAME_SKIP):
                # Step the environment
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Sum the squared rewards over this period
                accumulated_reward += reward

                done = terminated or truncated
                if done:
                    break

            # 3. Scale the Accumulated Squared Reward
            # Since we summed ~50 steps of massive squared penalties, we must scale aggressively.
            # A good target is to keep the reward roughly between -10 and 10.
            # If your typical reward per step is -300,000, 50 steps is -15,000,000.
            # We need to divide by a large factor, e.g., 1,000,000.
            scaled_reward = accumulated_reward / 100.0

            # 4. Store memory and Train
            # Important: We store the state BEFORE the skip and the state AFTER the skip
            agent.update(state, action, scaled_reward, next_obs, done)

            state = next_obs
            total_reward += accumulated_reward  # Keep track of real physics reward

        agent.decay_epsilon()  # decay epsilon after each step

        if episode % 10 == 0:
            print(
                f"Ep {episode} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.2f}"
            )

    # Save Model at the end
    if mode == "dqn":
        agent.save("dqn_final.pth")

    print("Training Complete.")
    return agent


if __name__ == "__main__":
    # Choose mode
    dqn_agent = train(mode="dqn")
