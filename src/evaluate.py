import numpy as np
from src.env import TrafficEnv
from src.baseline import FixedTimeAgent

def run_evaluation(agent, env, num_episodes=5):
    """
    Runs the agent in the environment for a set number of episodes.
    Returns the average reward per episode.
    """
    total_rewards = []

    print(f"--- Starting Evaluation ({num_episodes} episodes) ---")

    for i in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            # 1. Agent chooses action based on observation
            action = agent.act(obs)

            # 2. Environment takes step
            obs, reward, done, truncated, info = env.step(action)

            # 3. Track reward
            episode_reward += reward
            step_count += 1
            
            # Optional: Limit steps if env doesn't truncate automatically yet
            if step_count > 1000: 
                truncated = True

        total_rewards.append(episode_reward)
        print(f"Episode {i+1}: Total Reward = {episode_reward:.2f} (Steps: {step_count})")

    avg_reward = np.mean(total_rewards)
    print(f"\n>>> Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    # 1. Setup Environment
    env = TrafficEnv()

    # 2. Setup Baseline Agent (Switch every 30 steps)
    agent = FixedTimeAgent(cycle_duration=30)

    # 3. Run Benchmark
    run_evaluation(agent, env)