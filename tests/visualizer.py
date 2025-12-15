import matplotlib.pyplot as plt
import numpy as np

class TrafficVisualizer:
    def __init__(self, road_length=100):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.road_length = road_length
        
        # FIXED MAPPING:
        # Format: "LaneID": (Static_Axis_Val, Travel_Direction)
        self.lane_config = {
            "N2S": {"static_axis": -2, "direction": 1},  # x=-2, y = pos * 1
            "S2N": {"static_axis": 2,  "direction": -1}, # x= 2, y = pos * -1
            "E2W": {"static_axis": 2,  "direction": 1},  # y= 2, x = pos * 1  <-- WAS 2!
            "W2E": {"static_axis": -2, "direction": -1} # y=-2, x = pos * -1 <-- WAS -2!
        }
        
        self.colors = {"straight": "blue", "left": "orange", "right": "green"}

    def update(self, env):
        self.ax.clear()
        
        # 1. Setup Background (Roads)
        L = self.road_length
        # Draw the grey asphalt
        self.ax.fill_between([-4, 4], -L, L, color='gray', alpha=0.3)
        self.ax.fill_between([-L, L], -4, 4, color='gray', alpha=0.3)
        
        # 2. Draw Stop Lines / Traffic Lights
        # VISUAL TWEAK: Move the stop lines OUT from the center (0,0)
        # We draw them at +/- 4 meters instead of 0
        STOP_LINE_OFFSET = 4.0 
        
        ns_color = 'green' if env.current_phase == 0 else 'red'
        ew_color = 'green' if env.current_phase == 1 else 'red'
        
        if env.is_yellow:
            ns_color = 'yellow' if env.current_phase == 0 else 'red'
            ew_color = 'yellow' if env.current_phase == 1 else 'red'

        # NS Lights (Horizontal lines at y = +/- 4)
        self.ax.plot([-4, 4], [STOP_LINE_OFFSET, STOP_LINE_OFFSET], color=ns_color, linewidth=5, alpha=0.7)   # Top (N2S)
        self.ax.plot([-4, 4], [-STOP_LINE_OFFSET, -STOP_LINE_OFFSET], color=ns_color, linewidth=5, alpha=0.7) # Bottom (S2N)
        
        # EW Lights (Vertical lines at x = +/- 4)
        self.ax.plot([STOP_LINE_OFFSET, STOP_LINE_OFFSET], [-4, 4], color=ew_color, linewidth=5, alpha=0.7)   # Right (E2W)
        self.ax.plot([-STOP_LINE_OFFSET, -STOP_LINE_OFFSET], [-4, 4], color=ew_color, linewidth=5, alpha=0.7) # Left (W2E)

        # 3. Draw Cars with Offset
        for lane_id, cars in env.lanes.items():
            if lane_id not in self.lane_config: continue
            
            cfg = self.lane_config[lane_id]
            static_val = cfg["static_axis"]
            direction = cfg["direction"]
            
            x_vals = []
            y_vals = []
            c_vals = []
            
            for car in cars:
                pos = car["position"]
                intent = car.get("turn_intent", "straight")
                
                # VISUAL TWEAK: Add the offset to the position
                # If pos=0 (at physics stop line), visual pos becomes 4.0 (at visual stop line)
                visual_pos = pos + STOP_LINE_OFFSET
                
                # Coordinate Mapping
                if lane_id in ["N2S", "S2N"]:
                    x = static_val
                    y = visual_pos * direction 
                else:
                    x = visual_pos * direction 
                    y = static_val

                x_vals.append(x)
                y_vals.append(y)
                c_vals.append(self.colors.get(intent, "blue"))

            if x_vals:
                self.ax.scatter(x_vals, y_vals, c=c_vals, s=50, edgecolors='white', zorder=10)

        # 4. Final Formatting
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_aspect('equal') # Ensures 1 meter is 1 meter everywhere
        self.ax.set_title(f"Phase: {env.current_phase} | Time: {env.time_in_phase:.1f}s")
        self.ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.draw()
        plt.pause(0.001)