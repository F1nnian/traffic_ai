import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# Visualization Constants
# ==========================================
LANE_WIDTH = 4.0
ASPHALT_COLOR = "#4B4B4B"
MARKING_COLOR = "#E0E0E0"


class TrafficVisualizer:
    def __init__(self, road_length=100):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.road_length = road_length
        self.colors = {
            "straight": "#3498db",  # Blau
            "left": "#e67e22",  # Orange
            "right": "#2ecc71",  # Grün
        }

    def _get_lane_coordinates(self, lane_id, position, layout):
        """Berechnet Position des Autos (identisch zu vorher)."""
        direction = None
        index = 0
        for d, lanes in layout.items():
            if lane_id in lanes:
                direction = d
                index = lanes.index(lane_id)
                break

        if direction is None:
            return 0, 0

        # Offset (Mitte der Spur)
        # 0=Innen, 1=Außen
        center_gap = 0.2
        lateral_offset = center_gap + (index * LANE_WIDTH) + (LANE_WIDTH / 2)

        # WICHTIG: Das Auto-Position-System (Physics) nutzt 0.0 als Haltelinie.
        # Wir müssen das Auto visuell an die *richtige* Haltelinie verschieben.
        # Das machen wir aber dynamisch im update-Loop.
        # Hier geben wir erst mal nur den relativen Abstand zurück.

        return direction, lateral_offset, position

    def _calculate_road_geometry(self, layout):
        """Berechnet die Breiten der Straßenarme."""
        n_count = len(layout.get("N", ["A"]))
        s_count = len(layout.get("S", ["A"]))
        e_count = len(layout.get("E", ["A"]))
        w_count = len(layout.get("W", ["A"]))

        width_ns = max(n_count, s_count) * LANE_WIDTH
        width_ew = max(e_count, w_count) * LANE_WIDTH

        return width_ns, width_ew

    def _draw_road_network(self, width_ns, width_ew):
        """Zeichnet Asphalt und Markierungen."""
        L = self.road_length

        # Intersection Box (Die Grenzen der Kreuzung)
        limit_ns = width_ew  # Wie weit geht die horiz. Straße nach oben/unten?
        limit_ew = width_ns  # Wie weit geht die vert. Straße nach links/rechts?

        # 1. ASPHALT
        # Vertikal
        self.ax.add_patch(
            patches.Rectangle(
                (-width_ns, -L), 2 * width_ns, 2 * L, color=ASPHALT_COLOR, zorder=0
            )
        )
        # Horizontal
        self.ax.add_patch(
            patches.Rectangle(
                (-L, -width_ew), 2 * L, 2 * width_ew, color=ASPHALT_COLOR, zorder=0
            )
        )
        # Mitte (Fixiert Artefakte)
        self.ax.add_patch(
            patches.Rectangle(
                (-width_ns, -width_ew),
                2 * width_ns,
                2 * width_ew,
                color=ASPHALT_COLOR,
                zorder=0,
            )
        )

        # 2. MARKIERUNGEN
        def draw_dashed(x, y):
            self.ax.plot(
                x,
                y,
                color=MARKING_COLOR,
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                zorder=1,
            )

        def draw_solid_yellow(x, y):
            self.ax.plot(
                x, y, color="#F1C40F", linestyle="-", linewidth=2, alpha=0.9, zorder=1
            )

        # N/S Linien (Unterbrochen durch E/W Breite)
        draw_solid_yellow([0, 0], [limit_ns, L])  # Oben
        draw_solid_yellow([0, 0], [-L, -limit_ns])  # Unten

        # E/W Linien (Unterbrochen durch N/S Breite)
        draw_solid_yellow([limit_ew, L], [0, 0])  # Rechts
        draw_solid_yellow([-L, -limit_ew], [0, 0])  # Links

        # Dashed Lines für Spuren könnten hier ergänzt werden...
        # (Aus Platzgründen gekürzt, da die Logik identisch zu vorher ist)

    def _draw_traffic_lights(self, env, layout, width_ns, width_ew):
        """
        Zeichnet Ampeln pro Spur an der korrekten Position.
        """
        # Wo müssen die Autos halten?
        # N/S hält vor der Breite von E/W (+ kleiner Puffer)
        stop_y_pos = width_ew + 1.0
        stop_x_pos = width_ns + 1.0

        # Welche Lanes dürfen fahren?
        active_green_lanes = env.green_phases[env.current_phase]

        # Iteriere durch alle definierten Lanes im Layout
        for direction, lane_names in layout.items():
            for idx, lane_name in enumerate(lane_names):

                # 1. FARBE BESTIMMEN
                color = "#E74C3C"  # Rot (Default)

                if lane_name in active_green_lanes:
                    if env.is_yellow:
                        color = "#F1C40F"  # Gelb
                    else:
                        color = "#2ECC71"  # Grün (Hell)

                # 2. POSITION BERECHNEN
                # Lateral Offset (Seitlich)
                center_gap = 0.2
                lateral = center_gap + (idx * LANE_WIDTH) + (LANE_WIDTH / 2)

                # Koordinaten für den Strich
                if direction == "N":
                    # Oben, hält bei y = +stop_y_pos. Strich ist horizontal.
                    # x ist links der Achse (-lateral)
                    x_center = -lateral
                    y_center = stop_y_pos
                    self.ax.plot(
                        [x_center - 1.5, x_center + 1.5],
                        [y_center, y_center],
                        color=color,
                        lw=5,
                        zorder=5,
                    )

                elif direction == "S":
                    # Unten, hält bei y = -stop_y_pos
                    x_center = lateral
                    y_center = -stop_y_pos
                    self.ax.plot(
                        [x_center - 1.5, x_center + 1.5],
                        [y_center, y_center],
                        color=color,
                        lw=5,
                        zorder=5,
                    )

                elif direction == "E":
                    # Rechts, hält bei x = +stop_x_pos. Strich ist vertikal.
                    x_center = stop_x_pos
                    y_center = lateral
                    self.ax.plot(
                        [x_center, x_center],
                        [y_center - 1.5, y_center + 1.5],
                        color=color,
                        lw=5,
                        zorder=5,
                    )

                elif direction == "W":
                    # Links, hält bei x = -stop_x_pos
                    x_center = -stop_x_pos
                    y_center = -lateral
                    self.ax.plot(
                        [x_center, x_center],
                        [y_center - 1.5, y_center + 1.5],
                        color=color,
                        lw=5,
                        zorder=5,
                    )

    def update(self, env):
        self.ax.clear()
        self.ax.set_facecolor("#C8E6C9")

        # --- CALCULATE VIRTUAL TIME ---
        # Reuse the logic from your environment
        from src.config import MAX_STEPS_PER_EPISODE

        steps_per_hour = MAX_STEPS_PER_EPISODE / 24
        virtual_hour = int(env.total_steps // steps_per_hour)
        virtual_hour = min(virtual_hour, 23)

        # Format as string for the clock (e.g., "08:00")
        clock_str = f"{virtual_hour:02d}:00"

        layout = env.config["lanes"]

        # 1. Geometrie berechnen
        w_ns, w_ew = self._calculate_road_geometry(layout)

        # 2. Straße zeichnen
        self._draw_road_network(w_ns, w_ew)

        # 3. Ampeln zeichnen (NEU & DYNAMISCH)
        self._draw_traffic_lights(env, layout, w_ns, w_ew)

        # 4. Autos zeichnen
        # Puffer für Stop-Position (damit Auto an der Linie steht, nicht in der Kreuzung)
        stop_offset_ns = w_ew + 1.0
        stop_offset_ew = w_ns + 1.0

        for lane_id, cars in env.lanes.items():
            x_vals, y_vals, c_vals = [], [], []

            # Helper call
            res = self._get_lane_coordinates(lane_id, 0, layout)
            if res == (0, 0):
                continue
            direction, lateral, _ = res

            for car in cars:
                # Physics Position: 0.0 = Stop Line.
                # Visual Position: Stop Offset + Position
                pos = car["position"]

                if direction == "N":
                    x = -lateral
                    y = stop_offset_ns + pos
                elif direction == "S":
                    x = lateral
                    y = -(stop_offset_ns + pos)
                elif direction == "E":
                    x = stop_offset_ew + pos
                    y = lateral
                elif direction == "W":
                    x = -(stop_offset_ew + pos)
                    y = -lateral
                else:
                    x, y = 0, 0

                x_vals.append(x)
                y_vals.append(y)
                c_vals.append(
                    self.colors.get(car.get("turn_intent", "straight"), "blue")
                )

            if x_vals:
                self.ax.scatter(
                    x_vals, y_vals, c=c_vals, s=70, edgecolors="black", zorder=10
                )

        zoom = 100
        self.ax.set_xlim(-zoom, zoom)
        self.ax.set_ylim(-zoom, zoom)
        self.ax.set_aspect("equal")

        # Updated Title with Clock
        status = "YELLOW" if env.is_yellow else "GREEN"
        self.ax.set_title(
            f"TIME: {clock_str} | Phase: {env.current_phase} ({status})",
            fontsize=14,
            fontweight="bold",
        )

        # Optional: Add a stylized clock box in the top-right corner
        self.ax.text(
            zoom - 2,
            zoom - 5,
            f"VIRTUAL CLOCK\n{clock_str}",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            fontsize=10,
            ha="right",
            family="monospace",
        )

        plt.draw()
        plt.pause(0.001)
