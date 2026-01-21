import os
from src.scenarios import SCENARIOS

# 1. Simulation Settings
# Seconds per simulation step
DELTA_T = 0.1  # needs to be small in order for physics to work


# Length of the road in meters (per arm of the intersection)
ROAD_LENGTH = 100

# Traffic Light timings (in seconds)
# Minimum time a phase must stay active (to prevent rapid flickering)
MIN_PHASE_DURATION = 10
YELLOW_PHASE_DURATION = 4

# Traffic Generation (Poisson Distribution)
# Probability of a new car appearing per second
TRAFFIC_INTENSITY_NS = 0.5  # North-South is busy
TRAFFIC_INTENSITY_EW = 0.1  # East-West is quiet

# Car Physics
MAX_SPEED = 14  # m/s (approx 50 km/h)
ACCELERATION = 2.0  # m/s^2 (Standard car acceleration)
BRAKING_DECELERATION = 4.5  # m/s^2 (Standard braking)
SAFE_DISTANCE = 2.0  # Meters buffer between cars
MAX_TURN_SPEED = MAX_SPEED / 2.0  # m/s when turning at intersection

MIN_SAFE_TIME_GAP = 3.0  # seconds for turning

# Default Configuration
DEFAULT_CONFIG = SCENARIOS["SIMPLE"]

# 2. Reinforcement Learning Settings
# Q-Learning Parameters
ALPHA = 0.1  # Learning Rate
GAMMA = 0.99  # Discount Factor

# Exploration (Epsilon-Greedy)
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99  # How fast epsilon decreases per episode

# Discretization Buckets for Sum of Squared Waiting Times
SQ_WAIT_BUCKETS = [1, 900, 3600]


# 3. Training Settings
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 3600  # Max steps per episode (maps to 24 hours of simulation, for 14400 steps: 1s real time -> 1 minute simulated time)


# 4. File Paths
# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure these directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
