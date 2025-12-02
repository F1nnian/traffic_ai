import argparse
import time

from src.env import TrafficEnv
from src.baseline import FixedTimeAgent
from src.agent import QLearningAgent
from src.config import MODELS_DIR
import os


def run(agent_kind: str = "trained", steps: int = 1000, delay: float = 0.2):
    env = TrafficEnv()

    # Pick agent
    if agent_kind.lower() == "trained":
        try:
            # Build a QLearningAgent from env spaces and load trained table
            state_dim = tuple(env.observation_space.nvec.tolist())
            action_dim = int(env.action_space.n)
            agent = QLearningAgent(state_dim, action_dim, epsilon=0.0)
            agent.load(os.path.join(MODELS_DIR, "q_table.npy"))
        except Exception as e:
            print(f"Could not load trained agent ({e}). Falling back to FixedTime.")
            agent = FixedTimeAgent(cycle_duration=30)
    else:
        agent = FixedTimeAgent(cycle_duration=30)

    obs, _ = env.reset()

    for _ in range(steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render(mode="human")
        # Slow down the visualization so motions across the whole crossroad are visible
        if delay > 0:
            time.sleep(delay)
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo visualizer for TrafficEnv.")
    parser.add_argument("--agent", choices=["trained", "fixed"], default="trained",
                        help="Which agent to use: trained (Q-table) or fixed time")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to render")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between frames (e.g., 0.1..0.5)")
    args = parser.parse_args()

    run(agent_kind=args.agent, steps=args.steps, delay=args.delay)
