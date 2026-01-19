import src.config

src.config.MAX_STEPS_PER_EPISODE = (
    2400  # 10s real time -> 1 hour simulated time, so you can see the time switching
)
from src.env import TrafficEnv
import time
import matplotlib.pyplot as plt
import numpy as np
from tests.visualizer import TrafficVisualizer
from src.config import DELTA_T, ROAD_LENGTH, MAX_SPEED, SCENARIOS


def test_visual_demo():
    print(">>> STARTING VISUAL DEMO...")

    # 1. Setup Env with Turning Traffic | Options: SIMPLE, BIDIRECTIONAL, SHARED_LANES, DEDICATED_LANES, RUSH_HOUR
    env = TrafficEnv(config_name="RUSH_HOUR")

    env.reset()

    # 2. Init Visualizer
    viz = TrafficVisualizer(road_length=100)

    # 3. Loop
    try:
        for step in range(2400):
            # Switch lights every 50 steps
            action = 1 if step % 50 == 0 else 0

            env.step(action)

            # UPDATE THE VISUALIZATION
            viz.update(env)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopped by user")


def test_observation_evolution():
    print("\n>>> STARTING EVOLUTION TEST (RUSH_HOUR)...")

    # 1. Setup - Added include_intent=True
    env = TrafficEnv(
        config_name="DEDICATED_LANES",
        obs_mode="log",
        include_hour=True,
        include_queue=True,
        include_intent=True,
    )
    env.reset(seed=42)

    num_lanes = len(env.lane_ids)

    # 2. Dynamic Index Calculation
    # Structure: [Phase(1) | Wait(N) | Queue(N)? | Intent(N)? | Hour(1)?]

    current_pointer = 0

    # Phase
    phase_idx = current_pointer
    current_pointer += 1

    # Wait Times (Always present)
    wait_start_idx = current_pointer
    current_pointer += num_lanes

    # Check for Queue
    queue_start_idx = None
    if env.include_queue:
        queue_start_idx = current_pointer
        current_pointer += num_lanes

    # Check for Intent (NEW)
    intent_start_idx = None
    if env.include_intent:
        intent_start_idx = current_pointer
        current_pointer += num_lanes

    # Check for Hour
    hour_idx = None
    if env.include_hour:
        hour_idx = current_pointer
        current_pointer += 1

    snapshots = []

    # 3. Run Simulation
    for step in range(1001):
        # Force NS green (Action 0).
        obs, reward, terminated, truncated, info = env.step(0)

        if step % 20 == 0:
            snapshots.append((step, obs.copy()))

    # 4. Construct Dynamic Header
    header_parts = [f"{'Step':<6}", f"{'Phase':<6}"]

    # Always add Wait Columns
    for i in range(num_lanes):
        header_parts.append(f"{f'W_L{i}':<6}")

    # Conditionally add Queue Columns
    if env.include_queue:
        for i in range(num_lanes):
            header_parts.append(f"{f'Q_L{i}':<6}")

    # Conditionally add Intent Columns (NEW)
    if env.include_intent:
        for i in range(num_lanes):
            header_parts.append(f"{f'I_L{i}':<6}")

    # Conditionally add Hour Column
    if env.include_hour:
        header_parts.append(f"{'Hour':<6}")

    print(" | ".join(header_parts))
    print("-" * len(" | ".join(header_parts)))

    # 5. Print Dynamic Rows
    for step, o in snapshots:
        phase = o[phase_idx]

        # Start row
        row_parts = [f"{step:<6}", f"{phase:<6.1f}"]

        # Append Waits
        for i in range(num_lanes):
            val = o[wait_start_idx + i]
            row_parts.append(f"{val:<6.2f}")

        # Append Queues (if enabled)
        if queue_start_idx is not None:
            for i in range(num_lanes):
                val = o[queue_start_idx + i]
                row_parts.append(f"{val:<6.2f}")

        # Append Intent (if enabled)
        if intent_start_idx is not None:
            for i in range(num_lanes):
                val = o[intent_start_idx + i]
                row_parts.append(f"{val:<6.1f}")

        # Append Hour (if enabled)
        if hour_idx is not None:
            val = o[hour_idx]
            row_parts.append(f"{val:<6.2f}")

        print(" | ".join(row_parts))

    # 6. Assertions
    last_obs = snapshots[-1][1]
    first_obs = snapshots[0][1]

    print("\n>>> CHECKING LOGIC:")

    # Check Hour
    if env.include_hour:
        print(
            f"   [CHECK] Hour Evolution: {first_obs[hour_idx]:.2f} -> {last_obs[hour_idx]:.2f}"
        )
        assert last_obs[hour_idx] > first_obs[hour_idx], "Hour signal did not increase!"

    # Check Starvation (Last Lane)
    starved_lane_idx = num_lanes - 1
    starved_wait = last_obs[wait_start_idx + starved_lane_idx]

    print(
        f"   [CHECK] Starved Lane (L{starved_lane_idx}) Wait: {starved_wait:.2f} (Should be > 0)"
    )
    assert starved_wait > 0, "Starved lane wait time is zero!"

    if env.include_queue:
        starved_queue = last_obs[queue_start_idx + starved_lane_idx]
        print(
            f"   [CHECK] Starved Lane (L{starved_lane_idx}) Queue: {starved_queue:.2f} (Should be > 0)"
        )
        assert starved_queue > 0, "Starved lane queue is zero!"

    if env.include_intent:
        # Just check that we aren't crashing and values are within bounds
        starved_intent = last_obs[intent_start_idx + starved_lane_idx]
        print(
            f"   [CHECK] Starved Lane (L{starved_lane_idx}) Intent: {starved_intent:.1f} (Should be 0.0, 0.5, or 1.0)"
        )
        assert 0.0 <= starved_intent <= 1.0, "Intent value out of bounds!"

    print("-" * 70)
    print(">>> EVOLUTION TEST PASSED!")


if __name__ == "__main__":
    # test_visual_demo()
    test_observation_evolution()
