from src.env import TrafficEnv

def test_physics():
    env = TrafficEnv()
    obs, _ = env.reset()
    
    print(f"Initial Obs: {obs}")

    # Simulate 60 seconds    
    for i in range(60):
        action = 0 # Stay in one phase
        
        obs, reward, terminated, truncated, info = env.step(action)

        if info["ns_queue"] > 0 or info["ew_queue"] > 0:
            print(f"Step {i:02d} | Action: {action} | Obs: {obs} | Reward: {reward:.2f} | Queues: NS={info['ns_queue']} EW={info['ew_queue']}")


if __name__ == "__main__":
    test_physics()