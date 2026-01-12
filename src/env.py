import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from src.config import (
    DELTA_T,
    MIN_PHASE_DURATION,
    YELLOW_PHASE_DURATION,
    ROAD_LENGTH,
    MAX_SPEED,
    ACCELERATION,
    BRAKING_DECELERATION,
    SAFE_DISTANCE,
    MAX_STEPS_PER_EPISODE,
    SCENARIOS,
    DEFAULT_CONFIG,
    MIN_SAFE_TIME_GAP,
    MAX_TURN_SPEED,
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
        self.lanes = self.config["lanes"]
        self.lane_ids = []
        for lanes in self.lanes.values():
            self.lane_ids.extend(lanes)
        self.routes = self.config["routes"]
        self.green_phases = self.config["green_phases"]
        self.yield_map = self.config.get("yield_map", {})
        self.protected_phases = self.config.get("protected_phases", [])
        self.buckets = self.config["sq_wait_buckets"]
        self.enable_physics = self.config["enable_physics"]

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.MultiDiscrete(
            [len(self.green_phases)]
            + [24]
            + [len(self.buckets) + 1] * len(self.lane_ids)
        )

        self.current_phase = 0  # 0: NS Green, 1: EW Green
        self.total_steps = 0
        self.time_in_phase = 0  # Track seconds to enforce min duration/yellow
        self.is_yellow = False

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Reset internal state
        self.current_phase = 0
        self.total_steps = 0
        self.time_in_phase = 0
        self.is_yellow = False
        self.lanes = {name: [] for name in self.lane_ids}

        observation = self._get_obs(lane_waits=None)
        info = {}

        return observation, info

    def step(self, action):
        self.total_steps += 1

        num_phases = len(self.green_phases)

        # Handle Phase Switching
        if self.is_yellow:  # If in yellow phase
            self.time_in_phase += DELTA_T
            if self.time_in_phase >= YELLOW_PHASE_DURATION:
                self.current_phase = (self.current_phase + 1) % num_phases
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
            w = sum([c["wait_time"] ** 2 for c in self.lanes[lane_name]])
            total_sq_wait += w
            lane_waits.append(w)  # Collect for observation

        reward = -total_sq_wait
        observation = self._get_obs(lane_waits)

        terminated = self.total_steps >= MAX_STEPS_PER_EPISODE
        truncated = False

        info = {f"queue_{k}": len(v) for k, v in self.lanes.items()}

        return observation, reward, terminated, truncated, info

    def render(self):
        status = "yellow" if self.is_yellow else "green"
        print(
            f"Phase: {self.current_phase} ({status}) | Time: {self.time_in_phase:.1f}s"
        )

        # Dynamic print that handles ANY lane names
        stats = " | ".join([f"{k}: {len(v)}" for k, v in self.lanes.items()])
        print(stats)

    def _get_obs(self, lane_waits):
        # If called from reset(), create dummy zeros
        if lane_waits is None:
            lane_waits = [0] * len(self.lane_ids)

        # 1. Phase
        obs = [self.current_phase]

        steps_per_hour = MAX_STEPS_PER_EPISODE / 24
        virtual_hour = int(self.total_steps // steps_per_hour)
        obs.append(min(virtual_hour, 23))

        # 2. Buckets for each lane
        for w in lane_waits:
            bucket = np.digitize(w, self.buckets)
            obs.append(bucket)

        return np.array(obs, dtype=np.int32)

    def _get_current_intensity(self, route_data):
        # Calculate virtual hour (0-23)
        steps_per_hour = MAX_STEPS_PER_EPISODE / 24
        virtual_hour = int(self.total_steps // steps_per_hour)
        virtual_hour = min(virtual_hour, 23)

        # Extract schedule and find current intensity
        schedule = route_data["schedule"]
        current_i = schedule[0][1]

        for hour, intensity in schedule:
            if virtual_hour >= hour:
                current_i = intensity
            else:
                break

        return current_i

    def _spawn_cars(self):
        for route_id, data in self.routes.items():
            if self.np_random.random() < (self._get_current_intensity(data) * DELTA_T):
                self.lanes[data["lane"]].append(
                    {
                        "position": float(ROAD_LENGTH),
                        "speed": MAX_SPEED,
                        "wait_time": 0.0,
                        # Identity
                        "route_id": route_id,
                        "turn_intent": data["intent"],
                        # Physics
                        "length": 5.0,
                        "accel": ACCELERATION,
                        "decel": BRAKING_DECELERATION,
                        "max_speed": MAX_SPEED,
                    }
                )

    def _is_gap_safe(self, car, current_lane):
        car_route_id = car["route_id"]
        current_speed = car["speed"]
        # 1. Check Protected Phase (Green Arrow -> Always Safe)
        if self.current_phase in self.protected_phases:
            if current_lane in self.green_phases[self.current_phase]:
                return True

        # 2. Check Yield Map
        conflicting_routes = self.yield_map.get(car_route_id, [])
        if not conflicting_routes:
            return True

        dist_to_clear = 18.0  # Distance to clear intersection

        # Calculate time to clear intersection
        v_start = min(current_speed, MAX_TURN_SPEED)

        t_cross = (
            -v_start + math.sqrt(v_start**2 + 2 * ACCELERATION * dist_to_clear)
        ) / ACCELERATION

        required_gap = t_cross + 1.5

        # 3. Identify physical lanes to check (avoid duplicates)
        lanes_to_scan = set()
        for r_id in conflicting_routes:
            route_data = self.routes.get(r_id)
            if route_data:
                lanes_to_scan.add(route_data["lane"])

        # 4. Scan Lanes
        for lane_name in lanes_to_scan:
            all_cars = self.lanes.get(lane_name, [])

            # Look only at cars near the intersection
            approaching = [c for c in all_cars if c["position"] > -25.0]

            if not approaching:
                continue

            # Extract lead vehicle
            approaching.sort(key=lambda c: c["position"])
            lead_vehicle = approaching[0]

            threat_vehicle = None

            # If the lead vehicle is turning left
            if lead_vehicle["turn_intent"] == "left":
                # Check the shadow vehicle behind
                if len(approaching) > 1:
                    shadow_vehicle = approaching[1]
                    if shadow_vehicle["route_id"] in conflicting_routes:
                        threat_vehicle = shadow_vehicle
                else:
                    continue

            # If the lead vehicle is going straight or right
            elif lead_vehicle["route_id"] in conflicting_routes:
                threat_vehicle = lead_vehicle

            if threat_vehicle:

                if threat_vehicle["speed"] < 1.0 and threat_vehicle["position"] > -5.0:
                    continue

                # Calculate Time To Arrival
                if threat_vehicle["speed"] > 0.1:
                    t_pos = max(0.1, threat_vehicle["position"])
                    tta = t_pos / threat_vehicle["speed"]

                    if tta < required_gap:
                        return False

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
                is_green = lane_name in allowed_lanes

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
                        if not self._is_gap_safe(car, lane_name):
                            # Treat intersection as blocked wall
                            if dist_to_stop_line < dist_to_car_ahead:
                                dist_to_target = dist_to_stop_line
                                target_type = "yield"

                current_max_allowed = MAX_SPEED

                # Reduced speed for turning vehicles
                if turn_intent != "straight":
                    MAX_TURN_SPEED = MAX_SPEED * 0.5

                    if current_pos < 20.0:
                        current_max_allowed = MAX_TURN_SPEED

                # physics calculation
                if self.enable_physics:
                    # calculate required braking distance: d = v^2 / (2a)
                    if current_speed > 0:
                        braking_dist_needed = (current_speed**2) / (
                            2 * BRAKING_DECELERATION
                        )
                    else:
                        braking_dist_needed = 0

                    must_brake_for_obstacle = dist_to_target < (
                        braking_dist_needed + SAFE_DISTANCE
                    )

                    must_brake_for_turn = (current_speed > current_max_allowed) and (
                        current_pos < 20.0
                    )

                    if must_brake_for_obstacle or must_brake_for_turn:
                        # brake
                        new_speed = max(
                            0.0, current_speed - (BRAKING_DECELERATION * DELTA_T)
                        )
                    else:
                        # accelerate
                        # use current_max_allowed to respect turn speed
                        new_speed = min(
                            current_speed + (ACCELERATION * DELTA_T),
                            current_max_allowed,
                        )

                    move_dist = new_speed * DELTA_T

                    # make sure cars stop at obstacle/stop line
                    real_dist_to_obstacle = current_pos - pos_obstacle_ahead

                    if target_type in ["light", "yield"]:
                        real_dist_to_obstacle = dist_to_stop_line

                    if move_dist > (real_dist_to_obstacle - 0.1):
                        move_dist = max(0.0, real_dist_to_obstacle - 0.1)
                        new_speed = 0.0  # Force stop

                    car["speed"] = new_speed
                else:
                    # Fallback to old instant movement logic
                    dist_to_move = MAX_SPEED * DELTA_T
                    if dist_to_target <= 0:
                        move_dist = 0
                    else:
                        move_dist = min(dist_to_move, dist_to_target - 0.1)
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
