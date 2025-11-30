import os
from src.env import TrafficEnv
from src.agent import QLearningAgent
from src.config import (
    NUM_EPISODES,
    MAX_STEPS_PER_EPISODE,
    MODELS_DIR,
    EPSILON_MIN,
    EPSILON_DECAY,
)


def train():
    env = TrafficEnv()
    agent = QLearningAgent()

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        state = tuple(obs)
        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_obs)

            agent.update(state, action, reward, next_state)

            total_reward += float(reward)
            state = next_state

            if terminated or truncated:
                break

        # Epsilon decay per episode
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{NUM_EPISODES} - Total Reward: {total_reward:.3f} - Epsilon: {agent.epsilon:.3f}")

    # Save Q-table
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "q_table.npy")
    agent.save(save_path)
    print(f"Saved Q-table to {save_path}")

    return episode_rewards


if __name__ == "__main__":
    train()
