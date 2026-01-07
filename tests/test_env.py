from src.env import TrafficEnv
import time
import matplotlib.pyplot as plt
import numpy as np
from tests.visualizer import TrafficVisualizer
from src.config import DELTA_T, ROAD_LENGTH, MAX_SPEED, STEPS_PER_ACTION, SCENARIOS


def test_visual_demo():
    print(">>> STARTING VISUAL DEMO...")

    # 1. Setup Env with Turning Traffic | Options: SIMPLE, BIDIRECTIONAL, SHARED_LANES, DEDICATED_LANES
    env = TrafficEnv(config_name="DEDICATED_LANES")
    env.reset()

    # 2. Init Visualizer
    viz = TrafficVisualizer(road_length=100)

    # 3. Loop
    try:
        for step in range(5000):
            # Switch lights every 50 steps
            action = 1 if step % 50 == 0 else 0

            env.step(action)

            # UPDATE THE VISUALIZATION
            viz.update(env)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    # You can comment these out to run specific tests
    # test_physics()
    # test_asymmetry()
    # test_acceleration()
    # test_bidirectional_traffic()
    test_visual_demo()
