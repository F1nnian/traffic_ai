import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.config import (
    DELTA_T,
    MIN_PHASE_DURATION,
    YELLOW_PHASE_DURATION,
    SQ_WAIT_BUCKETS,
    TRAFFIC_INTENSITY_EW,
    TRAFFIC_INTENSITY_NS,
    ROAD_LENGTH,
    MAX_SPEED,
    ENABLE_PHYSICS,
    ACCELERATION,
    BRAKING_DECELERATION,
    SAFE_DISTANCE,
    STEPS_PER_ACTION
)


LANE_WIDTH = 4.0
LANE_OFFSET = 2.5
STOP_LINE_DISTANCE = 6.0


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

        # Lazy-render members (initialised on first render call)
        self._fig = None
        self._ax = None
        self._light_ns = None
        self._light_ew = None
        self._ns_scatter = None
        self._ew_scatter = None
        self._close_cid = None
        self._lane_offset = None

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

    def render(self, mode: str = "human"):
        """Visualise a four-way intersection with side-mounted traffic lights."""

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
        except Exception:
            status = "yellow" if self.is_yellow else "green"
            print(f"Phase: {self.current_phase} ({status}) | Time: {self.time_in_phase}s")
            print(f"Cars NS: {len(self.lanes['NS'])} | Cars EW: {len(self.lanes['EW'])}")
            return None

        lane_width = LANE_WIDTH
        lane_offset = LANE_OFFSET
        stop_line_thickness = 0.6
        light_radius = 1.2
        road_extent = ROAD_LENGTH
        road_half_span = lane_offset + (lane_width / 2.0)
        ns_lane_x = -lane_offset
        ew_lane_y = -lane_offset
        light_padding = lane_width * 1.2

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            ax = self._ax

            ax.set_xlim(-road_extent, road_extent)
            ax.set_ylim(-road_extent, road_extent)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_facecolor("#f0f0f0")

            # Draw orthogonal roads centred on both axes.
            ax.add_patch(Rectangle((-road_half_span, -road_extent), road_half_span * 2.0, road_extent * 2.0, color="#555555", zorder=0))
            ax.add_patch(Rectangle((-road_extent, -road_half_span), road_extent * 2.0, road_half_span * 2.0, color="#505050", alpha=0.95, zorder=1))
            ax.add_patch(Rectangle((-lane_width, -lane_width), lane_width * 2.0, lane_width * 2.0, color="#4a4a4a", zorder=2))

            # Stop lines positioned ahead of each incoming approach.
            ax.add_patch(Rectangle((-road_half_span, STOP_LINE_DISTANCE - stop_line_thickness / 2.0), road_half_span * 2.0, stop_line_thickness, color="white", zorder=4))
            ax.add_patch(Rectangle((-road_half_span, -STOP_LINE_DISTANCE - stop_line_thickness / 2.0), road_half_span * 2.0, stop_line_thickness, color="white", zorder=4))
            ax.add_patch(Rectangle((STOP_LINE_DISTANCE - stop_line_thickness / 2.0, -road_half_span), stop_line_thickness, road_half_span * 2.0, color="white", zorder=4))
            ax.add_patch(Rectangle((-STOP_LINE_DISTANCE - stop_line_thickness / 2.0, -road_half_span), stop_line_thickness, road_half_span * 2.0, color="white", zorder=4))

            # Traffic lights positioned to the side of each incoming lane
            self._light_ns = Circle((ns_lane_x - light_padding, STOP_LINE_DISTANCE + light_padding), light_radius, edgecolor="black", linewidth=2, zorder=6)
            self._light_ew = Circle((STOP_LINE_DISTANCE + light_padding, ew_lane_y - light_padding), light_radius, edgecolor="black", linewidth=2, zorder=6)
            ax.add_patch(self._light_ns)
            ax.add_patch(self._light_ew)

            # Car markers for the two travel axes.
            self._ns_scatter = ax.scatter([], [], s=90, c="#1976d2", marker="s", zorder=5)
            self._ew_scatter = ax.scatter([], [], s=90, c="#ef6c00", marker="s", zorder=5)

            self._lane_offset = lane_offset

            self._close_cid = self._fig.canvas.mpl_connect("close_event", self._handle_close)

            plt.show(block=False)
        else:
            lane_offset = self._lane_offset if self._lane_offset is not None else lane_offset
            ns_lane_x = -lane_offset
            ew_lane_y = -lane_offset
            light_padding = lane_width * 1.2

        if self.is_yellow:
            self._light_ns.set_color("yellow")
            self._light_ew.set_color("yellow")
        else:
            if self.current_phase == 0:
                self._light_ns.set_color("green")
                self._light_ew.set_color("red")
            else:
                self._light_ns.set_color("red")
                self._light_ew.set_color("green")

        ns_offsets = [(ns_lane_x, car["position"]) for car in self.lanes["NS"]]
        ew_offsets = [(car["position"], ew_lane_y) for car in self.lanes["EW"]]

        self._ns_scatter.set_offsets(np.array(ns_offsets, dtype=float) if ns_offsets else np.empty((0, 2)))
        self._ew_scatter.set_offsets(np.array(ew_offsets, dtype=float) if ew_offsets else np.empty((0, 2)))

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        if mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buffer = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buffer.reshape(h, w, 3)

        plt.pause(0.001)
        return None

    def close(self):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if self._fig is not None and plt.fignum_exists(self._fig.number):
            plt.close(self._fig)
        self._clear_render_handles()

    def _handle_close(self, _event):
        self._clear_render_handles()

    def _clear_render_handles(self):
        if self._fig is not None:
            try:
                if self._close_cid is not None:
                    self._fig.canvas.mpl_disconnect(self._close_cid)
            except Exception:
                pass

        self._fig = None
        self._ax = None
        self._light_ns = None
        self._light_ew = None
        self._ns_scatter = None
        self._ew_scatter = None
        self._close_cid = None
        self._lane_offset = None

    def _get_obs(self, sq_wait_ns, sq_wait_ew):
        # Current Phase
        p = self.current_phase

        # Discretize using buckets
        bucket_ns = np.digitize(sq_wait_ns, SQ_WAIT_BUCKETS)
        bucket_ew = np.digitize(sq_wait_ew, SQ_WAIT_BUCKETS)

        return np.array([p, bucket_ns, bucket_ew], dtype=np.int32)

    def _spawn_cars(self):
        # Try to spawn for North-South
        if self.np_random.random() < (TRAFFIC_INTENSITY_NS * DELTA_T):
            self.lanes["NS"].append({
                "position": float(ROAD_LENGTH),
                "wait_time": 0.0,
                "speed": MAX_SPEED
            })

        # Try to spawn for East-West
        if random.random() < (TRAFFIC_INTENSITY_EW * DELTA_T):
            self.lanes["EW"].append({
                "position": float(ROAD_LENGTH),
                "wait_time": 0.0,
                "speed": MAX_SPEED
            })

    def _move_cars(self):
        # Loop through both directions
        for lane_name in ["NS", "EW"]:
            lane = self.lanes[lane_name]
            
            # Determine traffic light status for this lane
            if self.is_yellow:
                is_green = False
            else:
                is_green = (self.current_phase == 0) if lane_name == "NS" else (self.current_phase == 1)

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
                dist_to_stop_line = current_pos - STOP_LINE_DISTANCE

                # determine which obstacle is relevant
                dist_to_target = dist_to_car_ahead
                target_type = "car"
                line_is_target = False

                # if light is not green, the stop line is a potential obstacle
                if not is_green:
                    if dist_to_stop_line > 0:
                        # if the line is closer than the car ahead, the line is the priority
                        if dist_to_stop_line < dist_to_car_ahead:
                            dist_to_target = dist_to_stop_line
                            target_type = "light"
                            line_is_target = True

                # physics calculation
                if ENABLE_PHYSICS:
                    # calculate required braking distance: d = v^2 / (2a)
                    if current_speed > 0:
                        braking_dist_needed = (current_speed**2) / (2 * BRAKING_DECELERATION)
                    else:
                        braking_dist_needed = 0

                    # dilemma zone (cant stop in time)
                    if line_is_target and dist_to_target < braking_dist_needed:
                        dist_to_target = dist_to_car_ahead
                        target_type = "car"
                        line_is_target = False
                    
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

                new_position = current_pos - move_dist
                if line_is_target and new_position < STOP_LINE_DISTANCE:
                    new_position = STOP_LINE_DISTANCE
                    car["speed"] = 0.0

                car["position"] = new_position

                # Update the obstacle ahead for the next car in the loop
                pos_obstacle_ahead = car["position"]

                # Remove cars that have cleared the intersection
                if car["position"] > -ROAD_LENGTH:
                    cars_to_keep.append(car)

            self.lanes[lane_name] = cars_to_keep