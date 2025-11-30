import os
from src.env import TrafficEnv
from src.agent import QLearningAgent
from src.config import NUM_EPISODES, MAX_STEPS_PER_EPISODE, MODELS_DIR


def main():
    env = TrafficEnv()
    agent = QLearningAgent()

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        state = tuple(obs)

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_obs)

            agent.update(state, action, reward, next_state)

            state = next_state
            if terminated or truncated:
                break

    # Ensure models dir exists and save Q-table
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "q_table.npy")
    agent.save(save_path)
    print(f"Saved Q-table to {save_path}")


if __name__ == "__main__":
    main()
