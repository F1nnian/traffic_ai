import os

# 1. Simulation Settings
# Seconds per simulation step
DELTA_T = 0.1  # needs to be small in order for physics to work
STEPS_PER_ACTION = 10  # AI acts every 10 steps (every second)


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
MAX_TURN_SPEED = 7.0  # m/s when turning at intersection

MIN_SAFE_TIME_GAP = 3.0  # seconds for turning

SCENARIOS = {
    "SIMPLE": {
        "description": "Original setup: 2 Lanes, Simple Physics, Asymmetric Traffic",
        "lanes": {"N": ["NS"], "E": ["EW"]},
        "green_phases": {
            0: ["NS"],  # Phase 0: NS Green
            1: ["EW"],  # Phase 1: EW Green
        },
        "routes": {
            "NS": {
                "i": 0.7,
                "lane": "NS",
                "intent": "straight",
            },  # North-South traffic intensity
            "EW": {
                "i": 0.3,
                "lane": "EW",
                "intent": "straight",
            },  # East-West traffic intensity
        },
        "enable_physics": False,
        # RL Buckets
        "sq_wait_buckets": [1, 1500, 5000],
    },
    # Placeholder for future issues
    "BIDIRECTIONAL": {
        "description": "4 Lanes (N2S, S2N, ...), Symmetric Traffic",
        "lanes": {"N": ["N2S"], "S": ["S2N"], "E": ["E2W"], "W": ["W2E"]},
        "routes": {
            "N2S": {"i": 0.3, "lane": "N2S", "intent": "straight"},
            "S2N": {"i": 0.3, "lane": "S2N", "intent": "straight"},
            "E2W": {"i": 0.3, "lane": "E2W", "intent": "straight"},
            "W2E": {"i": 0.3, "lane": "W2E", "intent": "straight"},
        },
        "green_phases": {
            0: ["N2S", "S2N"],  # Phase 0: North/South move
            1: ["E2W", "W2E"],  # Phase 1: East/West move
        },
        "enable_physics": True,
        "sq_wait_buckets": [1, 1500, 5000],  # Higher buckets for more cars
    },
    "SHARED_LANES": {
        "description": "4 Physical Lanes. Left turners share lane with straight traffic.",
        # Physical Lanes (Where cars exist)
        "lanes": {"N": ["N"], "S": ["S"], "E": ["E"], "W": ["W"]},
        # Logical Routes (Intensity, Physical Target Lane, Intent)
        "routes": {
            # North Approach
            "N2S": {"i": 0.4, "lane": "N", "intent": "straight"},
            "N2E": {"i": 0.1, "lane": "N", "intent": "left"},
            "N2W": {"i": 0.1, "lane": "N", "intent": "right"},
            # South Approach
            "S2N": {"i": 0.4, "lane": "S", "intent": "straight"},
            "S2W": {"i": 0.1, "lane": "S", "intent": "left"},
            "S2E": {"i": 0.1, "lane": "S", "intent": "right"},
            # East Approach
            "E2W": {"i": 0.3, "lane": "E", "intent": "straight"},
            "E2S": {"i": 0.1, "lane": "E", "intent": "left"},
            "E2N": {"i": 0.1, "lane": "E", "intent": "right"},
            # West Approach
            "W2E": {"i": 0.3, "lane": "W", "intent": "straight"},
            "W2N": {"i": 0.1, "lane": "W", "intent": "left"},
            "W2S": {"i": 0.1, "lane": "W", "intent": "right"},
        },
        # Conflict Map: Route X must yield to Route Y
        # (Left turners yield to oncoming straight/right traffic)
        "yield_map": {
            "N2E": ["S2N", "S2E"],
            "S2W": ["N2S", "N2W"],
            "E2S": ["W2E", "W2S"],
            "W2N": ["E2W", "E2N"],
        },
        # Traffic Light Phases (Control Physical Lanes)
        "green_phases": {0: ["N", "S"], 1: ["E", "W"]},
        "protected_phases": [],
        "enable_physics": True,
        "sq_wait_buckets": [1, 900, 3600],
    },
    "DEDICATED_LANES": {
        "description": "8 Physical Lanes. Exclusive Left Turn Lanes with Protected Phases.",
        # Physical Lanes (8 total)
        "lanes": {
            "N": ["N_Left", "N_Str"],
            "S": ["S_Left", "S_Str"],
            "E": ["E_Left", "E_Str"],
            "W": ["W_Left", "W_Str"],
        },
        # Logical Routes map to specific exclusive lanes
        "routes": {
            # North
            "N2S": {"i": 0.45, "lane": "N_Str", "intent": "straight"},
            "N2W": {"i": 0.10, "lane": "N_Str", "intent": "right"},
            "N2E": {"i": 0.15, "lane": "N_Left", "intent": "left"},
            # South
            "S2N": {"i": 0.45, "lane": "S_Str", "intent": "straight"},
            "S2E": {"i": 0.10, "lane": "S_Str", "intent": "right"},
            "S2W": {"i": 0.15, "lane": "S_Left", "intent": "left"},
            # East
            "E2W": {"i": 0.20, "lane": "E_Str", "intent": "straight"},
            "E2N": {"i": 0.10, "lane": "E_Str", "intent": "right"},
            "E2S": {"i": 0.10, "lane": "E_Left", "intent": "left"},
            # West
            "W2E": {"i": 0.20, "lane": "W_Str", "intent": "straight"},
            "W2S": {"i": 0.10, "lane": "W_Str", "intent": "right"},
            "W2N": {"i": 0.10, "lane": "W_Left", "intent": "left"},
        },
        # Conflict Map (Logic remains identical to Shared scenario)
        "yield_map": {
            "N2E": ["S2N", "S2E"],
            "S2W": ["N2S", "N2W"],
            "E2S": ["W2E", "W2S"],
            "W2N": ["E2W", "E2N"],
        },
        # 4-Phase Cycle including Protected Left Turns (Green Arrow)
        "green_phases": {
            0: ["N_Str", "S_Str", "N_Left", "S_Left"],  # Phase 0: NS General
            1: ["N_Left", "S_Left"],  # Phase 1: NS Protected Left
            2: ["E_Str", "W_Str", "E_Left", "W_Left"],  # Phase 2: EW General
            3: ["E_Left", "W_Left"],  # Phase 3: EW Protected Left
        },
        # In these phases, left turners ignore the yield map
        "protected_phases": [1, 3],
        "enable_physics": True,
        "sq_wait_buckets": [1, 900, 3600],
    },
}

# Default Configuration
DEFAULT_CONFIG = SCENARIOS["SIMPLE"]

# 2. Reinforcement Learning Settings
# Q-Learning Parameters
ALPHA = 0.1  # Learning Rate
GAMMA = 0.95  # Discount Factor

# Exploration (Epsilon-Greedy)
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # How fast epsilon decreases per episode

# Discretization Buckets for Sum of Squared Waiting Times
SQ_WAIT_BUCKETS = [1, 900, 3600]


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
