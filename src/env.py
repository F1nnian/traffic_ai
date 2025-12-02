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

        # Lazy-render state (matplotlib handles created on first render call)
        self._fig = None
        self._ax = None
        self._light_ns = None
        self._light_ew = None
        self._ns_scatter = None
        self._ew_scatter = None

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

    def render(self, mode: str = "human"):
        """Visualize the intersection using matplotlib.

        - human: shows/updates a window; returns None
        - rgb_array: returns an RGB numpy array of the current frame
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
        except Exception as e:
            # Fallback to text output if matplotlib isn't available
            status = "yellow" if self.is_yellow else "green"
            print(f"Phase: {self.current_phase} ({status}) | Time: {self.time_in_phase}s")
            print(f"Cars NS: {len(self.lanes['NS'])} | Cars EW: {len(self.lanes['EW'])}")
            return None

        # Initialize figure and artists lazily
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._ax.set_xlim(-ROAD_LENGTH, ROAD_LENGTH)
            self._ax.set_ylim(-ROAD_LENGTH, ROAD_LENGTH)
            self._ax.set_aspect('equal')
            self._ax.axis('off')

            # Roads
            road_w = 8
            horiz = Rectangle((-ROAD_LENGTH, -road_w/2), 2*ROAD_LENGTH, road_w, color='lightgray', zorder=0)
            vert = Rectangle((-road_w/2, -ROAD_LENGTH), road_w, 2*ROAD_LENGTH, color='lightgray', zorder=0)
            self._ax.add_patch(horiz)
            self._ax.add_patch(vert)

            # Lights
            self._light_ns = Circle((0, 14), 3, color='green')
            self._light_ew = Circle((14, 0), 3, color='red')
            self._ax.add_patch(self._light_ns)
            self._ax.add_patch(self._light_ew)

            # Car markers
            self._ns_scatter = self._ax.scatter([], [], s=40, c='blue')
            self._ew_scatter = self._ax.scatter([], [], s=40, c='orange')

        # Update lights
        if self.is_yellow:
            self._light_ns.set_color('yellow')
            self._light_ew.set_color('yellow')
        else:
            if self.current_phase == 0:
                self._light_ns.set_color('green')
                self._light_ew.set_color('red')
            else:
                self._light_ns.set_color('red')
                self._light_ew.set_color('green')

        # Update car positions on plot
        ns_xy = [(0.0, c["position"]) for c in self.lanes["NS"]]
        ew_xy = [(c["position"], 0.0) for c in self.lanes["EW"]]

        import numpy as _np
        if ns_xy:
            self._ns_scatter.set_offsets(_np.array(ns_xy, dtype=float))
        else:
            self._ns_scatter.set_offsets(_np.empty((0, 2)))
        if ew_xy:
            self._ew_scatter.set_offsets(_np.array(ew_xy, dtype=float))
        else:
            self._ew_scatter.set_offsets(_np.empty((0, 2)))

        # Draw/update
        self._fig.canvas.draw()

        if mode == "rgb_array":
            # Return RGB array
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            # Show interactively
            try:
                import matplotlib.pyplot as plt  # ensure pause is available
                plt.pause(0.001)
            except Exception:
                pass
            return None

    def close(self):
        """Close any open renderers."""
        try:
            import matplotlib.pyplot as plt
            if self._fig is not None:
                plt.close(self._fig)
        except Exception:
            pass
        finally:
            self._fig = None
            self._ax = None
            self._light_ns = None
            self._light_ew = None
            self._ns_scatter = None
            self._ew_scatter = None

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
