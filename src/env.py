import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.config import (
    DELTA_T,
    MIN_PHASE_DURATION,
    YELLOW_PHASE_DURATION,
    ROAD_LENGTH,
    MAX_SPEED,
    ACCELERATION,
    BRAKING_DECELERATION,
    SAFE_DISTANCE,
    STEPS_PER_ACTION,
    SCENARIOS,
    DEFAULT_CONFIG
)


class TrafficEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config_name="SIMPLE"):
        super(TrafficEnv, self).__init__()

        if config_name in SCENARIOS:
            self.config = SCENARIOS[config_name]
        else:
            print(f"Warning: Scenario '{config_name}' not found. Using DEFAULT.")
            self.config = DEFAULT_CONFIG

        # 2. Extract Config Values (for easier access)
        self.lane_ids = self.config["lanes"] # ["NS", "EW"] or ["N2S", "S2N", ...]
        self.green_phases = self.config["green_phases"]
        self.intensities = self.config["traffic_intensity"]
        self.buckets = self.config["sq_wait_buckets"]
        self.enable_physics = self.config["enable_physics"]

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.MultiDiscrete([2] + [4] * len(self.lane_ids))

        self.current_phase = 0  # 0: NS Green, 1: EW Green
        self.time_in_phase = 0  # Track seconds to enforce min duration/yellow
        self.is_yellow = False

        self.lanes = {name: [] for name in self.lane_ids}

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Reset internal state
        self.current_phase = 0
        self.time_in_phase = 0
        self.is_yellow = False
        self.lanes = {name: [] for name in self.lane_ids}

        observation = self._get_obs(lane_waits=None)
        info = {}

        return observation, info

    def step(self, action):
        # let physics run for 10 steps until next action
        for _ in range(STEPS_PER_ACTION):
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
        total_sq_wait = 0
        lane_waits = []
        
        for lane_name in self.lane_ids:
            w = sum([c["wait_time"]**2 for c in self.lanes[lane_name]])
            total_sq_wait += w
            lane_waits.append(w) # Collect for observation
            
        reward = -total_sq_wait
        observation = self._get_obs(lane_waits)
        
        info = {f"queue_{k}": len(v) for k,v in self.lanes.items()}

        return observation, reward, False, False, info

    def render(self):
        status = "yellow" if self.is_yellow else "green"
        print(f"Phase: {self.current_phase} ({status}) | Time: {self.time_in_phase:.1f}s")
        
        # Dynamic print that handles ANY lane names
        stats = " | ".join([f"{k}: {len(v)}" for k, v in self.lanes.items()])
        print(stats)
        
    def _get_obs(self, lane_waits):
    # If called from reset(), create dummy zeros
        if lane_waits is None:
            lane_waits = [0] * len(self.lane_ids)
            
        # 1. Phase
        obs = [self.current_phase]
        
        # 2. Buckets for each lane
        for w in lane_waits:
            bucket = np.digitize(w, self.buckets)
            obs.append(bucket)
            
        return np.array(obs, dtype=np.int32)

    def _spawn_cars(self):
        # Loop through dynamic lanes list
        for lane_name in self.lane_ids:
            # Get intensity for this specific lane
            intensity = self.intensities[lane_name]
            
            if self.np_random.random() < (intensity * DELTA_T):
                self.lanes[lane_name].append({
                    "position": float(ROAD_LENGTH),
                    "wait_time": 0.0,
                    "speed": MAX_SPEED if not self.enable_physics else MAX_SPEED 
                })

    def _move_cars(self):
# Check which lanes are allowed to move in current phase
        # Example: Phase 0 -> allowed=["NS"] (Simple) or ["N2S", "S2N"] (Bidirectional)
        allowed_lanes = self.green_phases[self.current_phase]
        
        # Loop through ALL lanes dynamically
        for lane_name in self.lane_ids:
            lane = self.lanes[lane_name]
            
            # Is this lane Green?
            if self.is_yellow:
                is_green = False
            else:
                is_green = (lane_name in allowed_lanes)

            # Sort (Cars with higher position values are further back)
            # process from closest to intersection (lowest pos) to furthest
            lane.sort(key=lambda c: c["position"])

            # Track position of the obstacle immediately ahead
            # Initialize effectively infinite (negative because cars move towards 0)
            # We use a large negative number effectively meaning "clear road ahead" for the first car
            pos_obstacle_ahead = -9999.0 

            cars_to_keep = []

            for car in lane:
                current_pos = car["position"]
                current_speed = car["speed"]
                
                # identify obstacle car
                dist_to_car_ahead = current_pos - pos_obstacle_ahead - SAFE_DISTANCE
                
                # identify obstacle stop line
                dist_to_stop_line = current_pos - 0.0

                # determine which obstacle is relevant
                dist_to_target = dist_to_car_ahead
                target_type = "car"

                # if light is not green, the stop line is a potential obstacle
                if not is_green:
                    if dist_to_stop_line > 0:
                        # if the line is closer than the car ahead, the line is the priority
                        if dist_to_stop_line < dist_to_car_ahead:
                            dist_to_target = dist_to_stop_line
                            target_type = "light"

                # physics calculation
                if self.enable_physics:
                    # calculate required braking distance: d = v^2 / (2a)
                    if current_speed > 0:
                        braking_dist_needed = (current_speed**2) / (2 * BRAKING_DECELERATION)
                    else:
                        braking_dist_needed = 0

                    # dilemma zone (cant stop in time)
                    if target_type == "light" and dist_to_target < braking_dist_needed:
                        dist_to_target = dist_to_car_ahead 
                    
                    if dist_to_target > (braking_dist_needed + SAFE_DISTANCE):
                        # accelerate
                        new_speed = current_speed + (ACCELERATION * DELTA_T)
                        new_speed = min(new_speed, MAX_SPEED)
                    else:
                        # brake
                        new_speed = current_speed - (BRAKING_DECELERATION * DELTA_T)
                        new_speed = max(0.0, new_speed)

                    car["speed"] = new_speed
                    move_dist = new_speed * DELTA_T

                else:
                    # Fallback to old instant movement logic
                    dist_to_move = MAX_SPEED * DELTA_T
                    if dist_to_target <= 0:
                        move_dist = 0
                    else:
                        move_dist = min(dist_to_move, dist_to_target)
                    car["speed"] = MAX_SPEED if move_dist > 0 else 0

                # update position
                if move_dist < 0.05 and dist_to_target < 2.0:
                    car["wait_time"] += DELTA_T

                car["position"] -= move_dist

                # Update the obstacle ahead for the next car in the loop
                pos_obstacle_ahead = car["position"]

                # Remove cars that have cleared the intersection
                if car["position"] > -10:
                    cars_to_keep.append(car)

            self.lanes[lane_name] = cars_to_keep