import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.config import (
    DELTA_T,
    MIN_PHASE_DURATION,
    YELLOW_PHASE_DURATION,
    SQ_WAIT_BUCKETS,
    TRAFFIC_INTENSITY,
    ROAD_LENGTH,
    MAX_SPEED
)


class TrafficEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        super(TrafficEnv, self).__init__()

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.MultiDiscrete([2, 4, 4])

        self.current_phase = 0  # 0: NS Green, 1: EW Green
        self.time_in_phase = 0  # Track seconds to enforce min duration/yellow
        self.is_yellow = False

        self.lanes = {"NS": [], "EW": []}

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Reset internal state
        self.current_phase = 0
        self.time_in_phase = 0
        self.is_yellow = False
        self.lanes = {"NS": [], "EW": []}

        observation = self._get_obs(0, 0)
        info = {}

        return observation, info

    def step(self, action):

        # Handle Phase Switching
        if self.is_yellow:  # If in yellow phase
            self.time_in_phase += DELTA_T
            if self.time_in_phase >= YELLOW_PHASE_DURATION:
                self.current_phase = 1 - self.current_phase  # Toggle 0 <-> 1
                self.is_yellow = False
                self.time_in_phase = 0
        else:
            # Normal Green Phase
            if action == 1:  # Agent wants to switch
                if self.time_in_phase >= MIN_PHASE_DURATION:
                    self.is_yellow = True
                    self.time_in_phase = 0
                else:
                    # Cannot switch yet
                    self.time_in_phase += DELTA_T
            else:
                self.time_in_phase += DELTA_T  # Agent wants to stay

        # Move and spawn cars
        self._spawn_cars()
        self._move_cars()

        # Calculate Reward
        ns_wait = sum([c["wait_time"]**2 for c in self.lanes["NS"]])
        ew_wait = sum([c["wait_time"]**2 for c in self.lanes["EW"]])
        
        reward = -(ns_wait + ew_wait)

        # Get Observation
        observation = self._get_obs(ns_wait, ew_wait)

        # Episode termination condition (Placeholder)
        terminated = False
        truncated = False
        info = {
            "ns_queue": len(self.lanes["NS"]),
            "ew_queue": len(self.lanes["EW"])
        }

        return observation, reward, terminated, truncated, info

    def render(self): # needs to be updated later for visualization (return cars and positions and stuff)
        status = "yellow" if self.is_yellow else "green"
        print(f"Phase: {self.current_phase} ({status}) | Time: {self.time_in_phase}s")
        print(f"Cars NS: {len(self.lanes['NS'])} | Cars EW: {len(self.lanes['EW'])}")

    def _get_obs(self, sq_wait_ns, sq_wait_ew):
        # Current Phase
        p = self.current_phase

        # Discretize using buckets
        bucket_ns = np.digitize(sq_wait_ns, SQ_WAIT_BUCKETS)
        bucket_ew = np.digitize(sq_wait_ew, SQ_WAIT_BUCKETS)

        return np.array([p, bucket_ns, bucket_ew], dtype=np.int32)

    def _spawn_cars(self):
        # Try to spawn for North-South
        if self.np_random.random() < (TRAFFIC_INTENSITY * DELTA_T):
            self.lanes["NS"].append({
                "position": float(ROAD_LENGTH),
                "wait_time": 0.0,
                "speed": MAX_SPEED
            })

        # Try to spawn for East-West
        if random.random() < (TRAFFIC_INTENSITY * DELTA_T):
            self.lanes["EW"].append({
                "position": float(ROAD_LENGTH),
                "wait_time": 0.0,
                "speed": MAX_SPEED
            })

    def _move_cars(self):
        # Minimum distance to keep between cars (meters)
        SAFE_DISTANCE = 2.0 

        # Loop through both directions
        for lane_name in ["NS", "EW"]:
            lane = self.lanes[lane_name]
            
            # Check if lane has green light
            if self.is_yellow:
                is_green = False
            else:
                if lane_name == "NS":
                    is_green = (self.current_phase == 0)
                else: # EW
                    is_green = (self.current_phase == 1)

            # Sort cars by position (Closest to intersection first)
            lane.sort(key=lambda c: c["position"])

            # Define the first obstacle
            next_obstacle_pos = 0.0 if not is_green else -9999.0

            cars_to_keep = []

            for car in lane:
                dist_to_move = MAX_SPEED * DELTA_T

                # Distance to the thing ahead (car or stop line)
                space_ahead = car["position"] - next_obstacle_pos - SAFE_DISTANCE

                if space_ahead <= 0:
                    move_dist = 0
                    car["wait_time"] += DELTA_T
                else:
                    move_dist = min(dist_to_move, space_ahead)
                    
                    # if move_dist < 0.1:
                    #      car["wait_time"] += DELTA_T

                car["position"] -= move_dist

                next_obstacle_pos = car["position"]

                # Remove cars that have left the intersection
                if car["position"] > -5:
                    cars_to_keep.append(car)

            # Update the main list with cars that are still here
            self.lanes[lane_name] = cars_to_keep
