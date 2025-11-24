from src.env import TrafficEnv

env = TrafficEnv()
obs, info = env.reset()

print("Initial Observation:", obs)

# Run for 5 steps with random actions
for _ in range(5):
    action = env.action_space.sample()  # Pick valid random action
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Obs: {obs}, Reward: {reward}")
