import os
from src.env import TrafficEnv
from src.agent import QLearningAgent
from src.config import (
    NUM_EPISODES,
    MAX_STEPS_PER_EPISODE,
    MODELS_DIR,
    EPSILON_MIN,
    EPSILON_DECAY,
    ALPHA,
    GAMMA,
    EPSILON_START,
)


def train():
    env = TrafficEnv()
    # Build agent using environment spaces to match agent constructor
    # observation_space.nvec -> array like [2,4,4]
    state_dim = tuple(env.observation_space.nvec.tolist())
    action_dim = int(env.action_space.n)
    agent = QLearningAgent(state_dim, action_dim, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        state = tuple(obs)
        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_obs)

            done = bool(terminated or truncated)
            # Agent.update expects (state, action, reward, next_state, done)
            agent.update(state, action, reward, next_state, done)

            total_reward += float(reward)
            state = next_state

            if done:
                break

        # Epsilon decay per episode
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        # Print progress every 100 episodes (and final episode)
        if (episode + 1) % 100 == 0 or (episode + 1) == NUM_EPISODES:
            print(f"Episode {episode+1}/{NUM_EPISODES}: Total Reward {total_reward:.3f}")

    # Save Q-table
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "q_table.npy")
    agent.save(save_path)
    print(f"Saved Q-table to {save_path}")

    return episode_rewards


if __name__ == "__main__":
    train()
