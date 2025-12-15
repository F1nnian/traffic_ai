from src.env import TrafficEnv
import matplotlib.pyplot as plt
import numpy as np
from src.config import (
    DELTA_T, 
    ROAD_LENGTH, 
    MAX_SPEED, 
    STEPS_PER_ACTION,
    SCENARIOS
)

def test_acceleration():
    print("=== TEST 1: Acceleration/Braking Physics ===")
    
    # 1. Setup Environment (Explicitly use SIMPLE with Physics)
    # Ensure physics is enabled in the config you are using
    config = SCENARIOS["SIMPLE"]
    config["enable_physics"] = True 
    
    env = TrafficEnv(config_name="SIMPLE")
    env.reset()

    # Disable random spawning so we only control our test car
    env._spawn_cars = lambda: None

    # 2. Force a Red Light Scenario
    # Phase 1 usually means EW is Green, so NS is RED.
    # We want NS to be Red to test braking.
    env.current_phase = 1 
    env.lanes["NS"] = []   # Clear random cars

    # 3. Manually spawn ONE car at the start of the road at full speed
    test_car = {
        "position": float(ROAD_LENGTH), 
        "wait_time": 0.0, 
        "speed": MAX_SPEED
    }
    env.lanes["NS"].append(test_car)

    # 4. Run Simulation
    history_speed = []
    history_pos = []
    time_steps = []

    # Run for 100 actions (100 * 10 steps = 1000 simulation ticks)
    for i in range(100):
        env.step(0) # Action 0 = Keep current phase (Keep NS Red)
        
        # Track the car (if it hasn't disappeared)
        if len(env.lanes["NS"]) > 0:
            car = env.lanes["NS"][0]
            history_speed.append(car["speed"])
            history_pos.append(car["position"])
            time_steps.append(i * DELTA_T * STEPS_PER_ACTION)
        else:
            print("Car has left the simulation (crossed 0).")
            break

    # 5. Plot Results
    print(f"Simulation finished. Plotting {len(time_steps)} data points...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Speed Plot
    ax1.plot(time_steps, history_speed, color='blue')
    ax1.set_title("Car Velocity approaching Red Light")
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True)
    ax1.axhline(y=MAX_SPEED, color='r', linestyle='--', label='Max Speed Limit')
    ax1.legend()

    # Position Plot
    ax2.plot(time_steps, history_pos, color='green')
    ax2.set_title("Car Position (0 = Stop Line)")
    ax2.set_ylabel("Distance to Stop Line (m)")
    ax2.axhline(y=0, color='red', linestyle='--', label='Stop Line')
    
    # Optional: Calculate where braking SHOULD have started (Visual Aid)
    # d = v^2 / 2a
    braking_dist = (MAX_SPEED**2) / (2 * 4.5) # 4.5 is hardcoded braking decel
    ax2.axhline(y=braking_dist, color='orange', linestyle=':', label='Theoretical Braking Dist')
    
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def test_physics():
    print("\n=== TEST 2: General Simulation Flow ===")
    env = TrafficEnv(config_name="SIMPLE")
    obs, _ = env.reset()
    
    print(f"Initial Obs: {obs}")

    # Simulate 60 steps (actions)
    for i in range(60):
        action = 0 # Stay in one phase
        
        obs, reward, terminated, truncated, info = env.step(action)

        # FIX: Access correct keys 'queue_NS' and 'queue_EW'
        ns_q = info.get("queue_NS", 0)
        ew_q = info.get("queue_EW", 0)

        if ns_q > 0 or ew_q > 0:
            print(f"Step {i:02d} | Action: {action} | Obs: {obs} | Reward: {reward:.2f} | Queues: NS={ns_q} EW={ew_q}")


def test_asymmetry(): 
    print("\n=== TEST 3: Asymmetric Traffic Generation ===")
    
    # FIX: Initialize Env first to get the config
    env = TrafficEnv(config_name="SIMPLE")
    
    # FIX: Access intensities from the env instance
    ns_intensity = env.intensities["NS"]
    ew_intensity = env.intensities["EW"]
    
    print(f"Simulation Setup: NS_Intensity={ns_intensity} vs EW_Intensity={ew_intensity}")

    env.reset()

    steps = 300
    print(f"Simulating {steps} steps...")

    ns_accumulated = 0
    ew_accumulated = 0

    for _ in range(steps):
        # Action 0 maintains phase 0 (NS Green). 
        # This keeps NS flowing (queue low) and EW blocked (queue high).
        # To test spawning, it's better to keep EVERYONE Red, but we can't easily do that.
        # Instead, we just switch periodically to let queues build up and clear.
        action = 1 if _ % 50 == 0 else 0
        
        _, _, _, _, info = env.step(action)
        
        # Accumulate the queue lengths as a proxy for traffic density
        ns_accumulated += info["queue_NS"]
        ew_accumulated += info["queue_EW"]

    print("-" * 30)
    print(f"Accumulated Queue Mass (Proxy for traffic volume):")
    print(f"🚗 NS Lane Score: {ns_accumulated}")
    print(f"🚗 EW Lane Score: {ew_accumulated}")
    print("-" * 30)

    # Note: Since NS is usually Green in this loop (Action 0), NS cars disappear faster!
    # So NS score might actually be LOWER than EW despite higher intensity.
    # To properly test asymmetry, we check the Config logic mostly.
    
    if ns_intensity > ew_intensity:
         print(f"✅ CONFIG CHECK: NS Intensity ({ns_intensity}) is correctly set higher than EW ({ew_intensity}).")
    else:
         print("❌ CONFIG CHECK: Asymmetry not configured correctly.")

def test_bidirectional_traffic():
    """
    Verifies that the TrafficEnv correctly handles bidirectional configuration
    without crashing and that cars spawn in all 4 lanes.
    """
    print("\n>>> STARTING BIDIRECTIONAL TEST...")

    # 1. Inject Test Configuration
    test_config_name = "TEST_BI_DIRECTIONAL"
    SCENARIOS[test_config_name] = {
        "lanes": ["N2S", "S2N", "E2W", "W2E"],
        "green_phases": [["N2S", "S2N"], ["E2W", "W2E"]],
        "traffic_intensity": {"N2S": 0.8, "S2N": 0.8, "E2W": 0.8, "W2E": 0.8}, # High intensity to ensure spawning
        "sq_wait_buckets": [10, 20, 30, 40], # Dummy buckets
        "enable_physics": False # Disable physics for faster testing
    }

    # 2. Initialize Environment
    try:
        env = TrafficEnv(config_name=test_config_name)
        obs, _ = env.reset()
    except Exception as e:
        print(f"❌ FAILED to initialize environment: {e}")
        return

    # 3. Run Simulation Loop
    cars_spawned = {lane: 0 for lane in env.lane_ids}
    
    for step in range(20):
        # Action 0: Keep phase (Phase 0: N2S/S2N Green)
        obs, reward, terminated, truncated, info = env.step(action=0)
        
        # Track max cars seen in each lane to verify spawning
        for lane in env.lane_ids:
            current_count = len(env.lanes[lane])
            if current_count > cars_spawned[lane]:
                cars_spawned[lane] = current_count

    # 4. Verify Results
    print(f"Max cars seen: {cars_spawned}")
    
    missing_lanes = [lane for lane, count in cars_spawned.items() if count == 0]
    
    if not missing_lanes:
        print("✅ SUCCESS: Traffic detected in all bidirectional lanes (N2S, S2N, E2W, W2E).")
    else:
        print(f"⚠️ FAILURE: No cars spawned in lanes: {missing_lanes}")
        print("Check your 'traffic_intensity' or '_spawn_cars' logic.")

if __name__ == "__main__":
    # You can comment these out to run specific tests
    # test_physics()
    # test_asymmetry()
    # test_acceleration()
    test_bidirectional_traffic()
