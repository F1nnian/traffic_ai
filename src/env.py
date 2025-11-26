import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.config import (
    DELTA_T,
    MIN_PHASE_DURATION,
    YELLOW_PHASE_DURATION,
    SQ_WAIT_BUCKETS,
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

        observation = self._get_obs()
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

        # Move and spawn cars (Placeholder)
        self._spawn_cars()
        self._move_cars()

        # Calculate Reward (Placeholder)
        reward = 0.0

        # Get Observation
        observation = self._get_obs()

        # Episode termination condition (Placeholder)
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        status = "yellow" if self.is_yellow else "green"
        print(f"Phase: {self.current_phase} ({status}) | Time: {self.time_in_phase}s")
        print(f"Cars NS: {len(self.lanes['NS'])} | Cars EW: {len(self.lanes['EW'])}")

    def _get_obs(self):
        # Current Phase
        p = self.current_phase

        # Calculate Sum of Squared Waiting Time (placeholder)
        sq_wait_ns_val = 0
        sq_wait_ew_val = 0

        # Discretize using buckets
        bucket_ns = np.digitize(sq_wait_ns_val, SQ_WAIT_BUCKETS)
        bucket_ew = np.digitize(sq_wait_ew_val, SQ_WAIT_BUCKETS)

        return np.array([p, bucket_ns, bucket_ew], dtype=np.int32)

    def _spawn_cars(self):
        pass

    def _move_cars(self):
        pass
