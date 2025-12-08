from src.env import TrafficEnv

import matplotlib.pyplot as plt
from src.config import DELTA_T, ROAD_LENGTH, MAX_SPEED

def test_acceleration():
    # 1. Setup Environment
    env = TrafficEnv()
    env.reset()

    env._spawn_cars = lambda: None

    # 2. Force a Red Light Scenario
    env.current_phase = 1  # 1 means EW is Green, so NS is RED
    env.lanes["NS"] = []   # Clear random cars

    # 3. Manually spawn ONE car at the start of the road at full speed
    test_car = {
        "position": float(ROAD_LENGTH), 
        "wait_time": 0.0, 
        "speed": MAX_SPEED
    }
    env.lanes["NS"].append(test_car)

    # 4. Run Simulation for 50 steps and record data
    history_speed = []
    history_pos = []
    time_steps = []

    for i in range(100):
        env.step(0) # Action 0 = Keep current phase (Keep NS Red)
        
        # Track the car (if it hasn't disappeared)
        if len(env.lanes["NS"]) > 0:
            car = env.lanes["NS"][0]
            history_speed.append(car["speed"])
            history_pos.append(car["position"])
            time_steps.append(i * DELTA_T)
        else:
            break

    # 5. Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Speed Plot
    ax1.plot(time_steps, history_speed, color='blue')
    ax1.set_title("Car Velocity approaching Red Light")
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True)
    # Add a line for MAX_SPEED to verify we didn't exceed it
    ax1.axhline(y=MAX_SPEED, color='r', linestyle='--', label='Max Speed Limit')

    # Position Plot
    ax2.plot(time_steps, history_pos, color='green')
    ax2.set_title("Car Position (0 = Stop Line)")
    ax2.set_ylabel("Distance to Stop Line (m)")
    ax2.axhline(y=0, color='red', linestyle='--', label='Stop Line')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

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
    test_acceleration()
