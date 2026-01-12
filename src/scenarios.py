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
                "schedule": [0, 0.7],
                "lane": "NS",
                "intent": "straight",
            },  # North-South traffic intensity
            "EW": {
                "schedule": [0, 0.3],
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
            "N2S": {"schedule": [0, 0.3], "lane": "N2S", "intent": "straight"},
            "S2N": {"schedule": [0, 0.3], "lane": "S2N", "intent": "straight"},
            "E2W": {"schedule": [0, 0.3], "lane": "E2W", "intent": "straight"},
            "W2E": {"schedule": [0, 0.3], "lane": "W2E", "intent": "straight"},
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
            "N2S": {"schedule": [0, 0.4], "lane": "N", "intent": "straight"},
            "N2E": {"schedule": [0, 0.1], "lane": "N", "intent": "left"},
            "N2W": {"schedule": [0, 0.1], "lane": "N", "intent": "right"},
            # South Approach
            "S2N": {"schedule": [0, 0.4], "lane": "S", "intent": "straight"},
            "S2W": {"schedule": [0, 0.1], "lane": "S", "intent": "left"},
            "S2E": {"schedule": [0, 0.1], "lane": "S", "intent": "right"},
            # East Approach
            "E2W": {"schedule": [0, 0.3], "lane": "E", "intent": "straight"},
            "E2S": {"schedule": [0, 0.1], "lane": "E", "intent": "left"},
            "E2N": {"schedule": [0, 0.1], "lane": "E", "intent": "right"},
            # West Approach
            "W2E": {"schedule": [0, 0.3], "lane": "W", "intent": "straight"},
            "W2N": {"schedule": [0, 0.1], "lane": "W", "intent": "left"},
            "W2S": {"schedule": [0, 0.1], "lane": "W", "intent": "right"},
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
            "N2S": {"schedule": [0, 0.45], "lane": "N_Str", "intent": "straight"},
            "N2W": {"schedule": [0, 0.10], "lane": "N_Str", "intent": "right"},
            "N2E": {"schedule": [0, 0.15], "lane": "N_Left", "intent": "left"},
            # South
            "S2N": {"schedule": [0, 0.45], "lane": "S_Str", "intent": "straight"},
            "S2E": {"schedule": [0, 0.10], "lane": "S_Str", "intent": "right"},
            "S2W": {"schedule": [0, 0.15], "lane": "S_Left", "intent": "left"},
            # East
            "E2W": {"schedule": [0, 0.20], "lane": "E_Str", "intent": "straight"},
            "E2N": {"schedule": [0, 0.10], "lane": "E_Str", "intent": "right"},
            "E2S": {"schedule": [0, 0.10], "lane": "E_Left", "intent": "left"},
            # West
            "W2E": {"schedule": [0, 0.20], "lane": "W_Str", "intent": "straight"},
            "W2S": {"schedule": [0, 0.10], "lane": "W_Str", "intent": "right"},
            "W2N": {"schedule": [0, 0.10], "lane": "W_Left", "intent": "left"},
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
