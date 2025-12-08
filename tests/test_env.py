from src.env import TrafficEnv

import matplotlib.pyplot as plt
from src.config import DELTA_T, ROAD_LENGTH, MAX_SPEED
import src.config as config_module

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


def test_asymmetry():    
    print(f"Simulation Setup: NS_Intensity={config_module.TRAFFIC_INTENSITY_NS} vs EW_Intensity={config_module.TRAFFIC_INTENSITY_EW}")

    env = TrafficEnv()
    env.reset()

    # 3. Simulation laufen lassen
    # Wir brauchen ca. 200-500 Steps, damit der Zufall sich ausgleicht
    steps = 300
    print(f"Simulating {steps} steps (approx {steps * config_module.STEPS_PER_ACTION * config_module.DELTA_T:.0f} seconds)...")

    ns_count = len(env.lanes["NS"])
    ew_count = len(env.lanes["EW"])

    for _ in range(steps):
        # Action 0 = Phase beibehalten. 
        # Wir lassen einfach alles auflaufen, um die Spawn-Raten zu sehen.
        env.step(1) 
        ns_count += len(env.lanes["NS"])
        ew_count += len(env.lanes["EW"])



    print("-" * 30)
    print(f"Final Lane Counts:")
    print(f"🚗 NS Lane: {ns_count} cars")
    print(f"🚗 EW Lane: {ew_count} cars")
    print("-" * 30)

    # 5. Check
    if ns_count > (ew_count * 4):
        print("✅ SUCCESS: NS traffic is dominantly higher. Asymmetry logic works.")
    elif ns_count > ew_count:
        print("⚠️ WARNING: NS is higher, but not by a massive margin. Check randomness.")
    else:
        print("❌ FAILURE: NS traffic is not higher than EW. Check your _spawn_cars logic.")

if __name__ == "__main__":
    # test_physics()
    # test_acceleration()
    test_asymmetry()
