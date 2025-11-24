import os

# 1. Simulation Settings
# Seconds per simulation step
DELTA_T = 1

# Length of the road in meters (per arm of the intersection)
ROAD_LENGTH = 100

# Traffic Light timings (in seconds)
# Minimum time a phase must stay active (to prevent rapid flickering)
MIN_PHASE_DURATION = 10
YELLOW_PHASE_DURATION = 4

# Traffic Generation (Poisson Distribution)
# Probability of a new car appearing per second
TRAFFIC_INTENSITY = 0.3

# Car Physics
MAX_SPEED = 14  # m/s (approx 50 km/h)
ACCELERATION = 2  # m/s^2


# 2. Reinforcement Learning Settings
# Q-Learning Parameters
ALPHA = 0.1  # Learning Rate
GAMMA = 0.95  # Discount Factor

# Exploration (Epsilon-Greedy)
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # How fast epsilon decreases per episode

# Discretization Buckets for Sum of Squared Waiting Times
SQ_WAIT_BUCKETS = [1, 101, 401]


# 3. Training Settings
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 300  # Max steps per episode (5 minutes of simulation)


# 4. File Paths
# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure these directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
