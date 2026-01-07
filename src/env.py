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
    DEFAULT_CONFIG,
    MIN_SAFE_TIME_GAP
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
        self.turning_ratios = self.config.get("turning_ratios", {})

        self.opposing_lane_map = self.config.get("opposing_lanes", {})

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

                # default to no turning
                ratios = self.turning_ratios.get(lane_name, {"left": 0.0, "straight": 1.0, "right": 0.0})
                
                turn_intent = self.np_random.choice(
                    ["left", "straight", "right"],
                    p=[ratios["left"], ratios["straight"], ratios["right"]]
                )

                self.lanes[lane_name].append({
                    "position": float(ROAD_LENGTH),
                    "wait_time": 0.0,
                    "speed": MAX_SPEED,
                    "turn_intent": turn_intent
                })

    def _is_gap_safe(self, lane_name):
        opposing_id = self.opposing_lane_map.get(lane_name)
        
        if opposing_id is not None and opposing_id in self.lanes:
            opposing_cars = self.lanes[opposing_id]
            
            # 1. Filter for cars that are relevant (not passed yet)
            # We use > -5 to catch cars just inside the intersection too
            approaching_cars = [c for c in opposing_cars if c["position"] > -5.0]
            
            # 2. If no cars, it's safe
            if not approaching_cars:
                return True
                
            # 3. Sort by position (closest to intersection first)
            approaching_cars.sort(key=lambda c: c["position"])
            
            # 4. GET THE LEAD CAR
            lead_car = approaching_cars[0]
            
            lead_pos = lead_car["position"]
            lead_speed = lead_car["speed"]
            lead_intent = lead_car.get("turn_intent", "straight")
            
            # --- THE LOGIC FIX ---
            # If the LEAD car is turning left, he blocks his own lane.
            # We can proceed safely (simultaneous left turn).
            if lead_intent == "left":
                return True
            
            # If the LEAD car is Straight/Right, we check safety physics:
            
            # A. Is he far away? (Time to Arrival)
            if lead_speed > 0.1:
                tta = lead_pos / lead_speed
                if tta < MIN_SAFE_TIME_GAP:
                    return False # Too fast, too close
            
            # B. Is he close and stopped? (e.g. waiting at line)
            else:
                if lead_pos < 15.0:
                    return False # He's right there waiting to go straight
                    
        return True
    
    def _move_cars(self):
        # check which lanes are currently allowed to move
        allowed_lanes = self.green_phases[self.current_phase]
        
        # loop through all lanes
        for lane_name in self.lane_ids:
            lane = self.lanes[lane_name]
            
            # is lane green?
            if self.is_yellow:
                is_green = False
            else:
                is_green = (lane_name in allowed_lanes)

            # sort cars by position
            lane.sort(key=lambda c: c["position"])

            pos_obstacle_ahead = -9999.0 

            cars_to_keep = []

            for car in lane:
                current_pos = car["position"]
                current_speed = car["speed"]
                turn_intent = car["turn_intent"]
                
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

                # Only check if Green, Turning Left, and near intersection
                if is_green and turn_intent == "left":
                    if 0 < dist_to_stop_line < 10.0:
                        if not self._is_gap_safe(lane_name):
                            # Treat intersection as blocked wall
                            if dist_to_stop_line < dist_to_car_ahead:
                                dist_to_target = dist_to_stop_line
                                target_type = "yield"

                # physics calculation
                if self.enable_physics:
                    # calculate required braking distance: d = v^2 / (2a)
                    if current_speed > 0:
                        braking_dist_needed = (current_speed**2) / (2 * BRAKING_DECELERATION)
                    else:
                        braking_dist_needed = 0

                    # dilemma zone (cant stop in time)
                    if target_type in ["light", "yield"] and dist_to_target < braking_dist_needed:
                        dist_to_target = dist_to_car_ahead 
                    
                    if dist_to_target > (braking_dist_needed + SAFE_DISTANCE):
                        # accelerate
                        new_speed = current_speed + (ACCELERATION * DELTA_T)
                        new_speed = min(new_speed, MAX_SPEED)

                        if current_pos < 10 and turn_intent != "straight":
                             new_speed = min(new_speed, MAX_SPEED * 0.6)
                    else:
                        # brake
                        new_speed = current_speed - (BRAKING_DECELERATION * DELTA_T)
                        new_speed = max(0.0, new_speed)

                    move_dist = new_speed * DELTA_T
                    
                    # make sure cars stop at obstacle/stop line
                    real_dist_to_obstacle = current_pos - pos_obstacle_ahead
                    
                    if target_type in ["light", "yield"]:
                        real_dist_to_obstacle = dist_to_stop_line

                    if move_dist > (real_dist_to_obstacle - 0.1):
                        move_dist = max(0.0, real_dist_to_obstacle - 0.1)
                        new_speed = 0.0 # Force stop

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