from manim import *
import numpy as np

# ==================================================
# SCENE 1 — 2D CURL AND OUT-OF-PLANE VECTOR
# Narration: ~15 seconds
# Total runtime: ~18 seconds
# ==================================================

class Scene1_2DCurl(ThreeDScene):
    def construct(self):
        # Camera setup - start top-down (2D view looking down at x-y plane)
        self.set_camera_orientation(phi=0*DEGREES, theta=-90*DEGREES, zoom=2.5)

        # Create 2D axes - ZOOMED OUT with more marks
        axes = ThreeDAxes(
            x_range=[-4, 4, 0.5],  # More axis marks (every 0.5 units)
            y_range=[-4, 4, 0.5],
            z_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            z_length=2
        )
        # Make axis arrow tips smaller
        for axis in axes:
            if axis.tip:
                axis.tip.scale(0.5)

        # Define 2D vector field with 2-3 vortices (adjusted for zoomed out view)
        vortex_centers = [
            np.array([-1.5, 1.5, 0.0]),   # Top-left vortex (strong)
            np.array([1.8, -1.0, 0.0]),   # Right vortex (medium)
            np.array([-0.5, -2.0, 0.0])   # Bottom vortex (weak)
        ]
        vortex_strengths = [1.5, 1.0, 0.5]

        def v_field(point):
            x, y, z = point
            velocity = np.array([0.0, 0.0, 0.0])

            for center, strength in zip(vortex_centers, vortex_strengths):
                r = point - center
                d = np.linalg.norm(r[:2]) + 0.3  # Avoid singularity

                # Tangential flow (perpendicular to radius in 2D)
                v_tangent = np.array([-r[1], r[0], 0.0]) * strength / (d**2)
                velocity += v_tangent

            return velocity

        # Curl magnitude function (for visualization)
        def curl_magnitude(point):
            eps = 0.1
            x, y, z = point

            # Approximate curl using finite differences
            vx_y = (v_field(point + np.array([0, eps, 0]))[0] -
                   v_field(point - np.array([0, eps, 0]))[0]) / (2*eps)
            vy_x = (v_field(point + np.array([eps, 0, 0]))[1] -
                   v_field(point - np.array([eps, 0, 0]))[1]) / (2*eps)

            curl_z = vy_x - vx_y
            return curl_z

        # Create vector field arrows with HIGHER DENSITY
        velocity_field = VGroup()
        for x in np.arange(-3.5, 3.6, 0.3):  # Adjusted for zoomed out view
            for y in np.arange(-3.5, 3.6, 0.3):
                point = axes.c2p(x, y, 0)
                v = v_field(np.array([x, y, 0]))

                v_norm = np.linalg.norm(v)
                if v_norm > 0.01:
                    # Scale arrows for visibility
                    v_scaled = v * 0.2 / max(v_norm, 0.25)
                    arrow = Arrow(
                        start=point,
                        end=point + v_scaled,
                        color=BLUE,
                        buff=0,
                        stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2
                    )
                    velocity_field.add(arrow)

        # Animation: Create axes and vector field (0-3s)
        # Narration: "Imagine a simple 2D vector field."
        self.play(Create(axes, run_time=1.5))
        self.wait(0.5)  # Brief pause

        # Add "velocity field u" label (BLUE to match vector field)
        velocity_label = MathTex(r"\text{velocity field } \vec{u}", font_size=36, color=BLUE)
        velocity_label.to_edge(UP)
        self.add_fixed_in_frame_mobjects(velocity_label)

        self.play(
            FadeIn(velocity_field, lag_ratio=0.05, run_time=2),
            Write(velocity_label, run_time=0.8)
        )
        self.wait(0.5)
        self.play(FadeOut(velocity_label), run_time=0.7)
        self.wait(0.5)  # Hold to show the field

        # Create SMALLER probe dot at center of MEDIUM vortex (to keep curl < 10)
        # Narration: "At any point in that field..."
        probe_center = vortex_centers[1]  # Medium vortex (right side)
        probe_dot = Dot(color=YELLOW, radius=0.08)  # Smaller dot
        probe_dot.move_to(axes.c2p(*probe_center[:2], 0))

        # SMALLER sampling circle around dot
        sampling_circle = Circle(radius=0.25, color=WHITE, stroke_width=2)  # Smaller circle
        sampling_circle.move_to(probe_dot.get_center())

        self.play(
            FadeIn(probe_dot),
            Create(sampling_circle),
            run_time=1
        )
        self.wait(0.5)

        # Make sampling circle follow dot with updater
        sampling_circle.add_updater(lambda m: m.move_to(probe_dot.get_center()))

        # Add tangential velocity vectors on circle border showing local curl
        def create_circle_tangent_arrows():
            arrows = VGroup()
            circle_radius = sampling_circle.radius
            num_arrows = 8
            for i in range(num_arrows):
                angle = i * TAU / num_arrows
                # Position on circle
                circle_point = probe_dot.get_center() + circle_radius * np.array([np.cos(angle), np.sin(angle), 0])
                # Get velocity at that point
                point_3d = axes.p2c(circle_point)
                v = v_field(point_3d)
                v_norm = np.linalg.norm(v[:2])
                if v_norm > 0.01:
                    v_scaled = v * 0.15 / max(v_norm, 0.2)
                    arrow = Arrow(
                        start=circle_point,
                        end=circle_point + v_scaled,
                        color=ORANGE,  # Changed to orange
                        buff=0,
                        stroke_width=3,  # Thicker arrows
                        max_tip_length_to_length_ratio=0.25
                    )
                    arrows.add(arrow)
            return arrows

        circle_arrows = create_circle_tangent_arrows()
        circle_arrows.add_updater(lambda m: m.become(create_circle_tangent_arrows()))

        self.play(FadeIn(circle_arrows), run_time=0.8)

        # Add curl magnitude display during 2D circular motion with 3b1b-style background
        # Fixed position at LEFT side (-3, 3) in axis coordinates
        def create_curl_display_2d():
            pos_3d = axes.p2c(probe_dot.get_center())
            curl_val = curl_magnitude(pos_3d)  # Keep sign (can be negative)

            # Create text with larger font
            text = MathTex(
                r"\nabla \times \vec{u} = " + f"{curl_val:.2f}",
                font_size=40
            )

            # Fixed position at LEFT side (-3, 3) in axis coordinates
            text.move_to(axes.c2p(-3, 3, 0))

            # Add 3b1b-style background rectangle (dark with opacity)
            # Increased buff to accommodate varying text widths (single/double digits, minus signs)
            bg_rect = BackgroundRectangle(
                text,
                color=BLACK,
                fill_opacity=0.8,
                buff=0.3  # Increased from 0.15 to 0.3
            )

            return VGroup(bg_rect, text)

        curl_display_2d = always_redraw(create_curl_display_2d)
        self.add_fixed_in_frame_mobjects(curl_display_2d)
        self.play(FadeIn(curl_display_2d), run_time=0.5)

        # Circle motion - SLOWER and returns to starting position
        # Narration: "the curl tells us how strongly the flow spins around that point"
        circle_radius = 0.3
        circular_path = ParametricFunction(
            lambda t: axes.c2p(
                probe_center[0] + circle_radius * np.cos(t),
                probe_center[1] + circle_radius * np.sin(t),
                0
            ),
            t_range=[0, 2*TAU]  # Complete 2 full circles to return to start
        )

        self.play(
            MoveAlongPath(probe_dot, circular_path, run_time=6, rate_func=linear)  # Slower (6s instead of 4s)
        )
        self.wait(0.5)

        # Keep the curl display during the rest of the animation (already added)
        # Keep circle arrows visible during 3D visualization (don't fade out)
        # circle_arrows.clear_updaters()  # Keep updater active

        # Pause the dot and create curl vector pointing toward camera
        # Narration: "And since curl is actually a 3D operation..."
        current_pos = probe_dot.get_center()
        curl_mag = abs(curl_magnitude(axes.p2c(current_pos)))

        # Create curl vector in 3D space pointing along z-axis
        # This will point toward camera in top-down view
        curl_length = min(curl_mag * 5, 3.0)  # Very sensitive scaling

        # Use axes coordinate system to create proper 3D arrow
        z_direction = axes.c2p(0, 0, curl_length) - axes.c2p(0, 0, 0)
        curl_vector = Arrow3D(
            start=current_pos,
            end=current_pos + z_direction,
            color=YELLOW,  # Changed to yellow
            thickness=0.02,
            height=0.3,
            base_radius=0.08
        )

        # Grow the curl vector (7-9s)
        # Narration: "it returns a vector perpendicular to the plane"
        self.play(Create(curl_vector), run_time=1.5)
        self.wait(0.5)

        # Create a new curl display for 3D phase (positioned on LEFT side)
        def create_curl_display_3d():
            pos_3d = axes.p2c(probe_dot.get_center())
            curl_val = curl_magnitude(pos_3d)  # Keep sign (can be negative)

            # Create text with larger font
            text = MathTex(
                r"\nabla \times \vec{u} = " + f"{curl_val:.2f}",
                font_size=40
            )

            # Position on the LEFT side to stay out of the way
            text.move_to(axes.c2p(-3.5, 3.5, 0))

            # Add 3b1b-style background rectangle (dark with opacity)
            # Increased buff to accommodate varying text widths (single/double digits, minus signs)
            bg_rect = BackgroundRectangle(
                text,
                color=BLACK,
                fill_opacity=0.8,
                buff=0.3  # Increased from 0.15 to 0.3
            )

            return VGroup(bg_rect, text)

        # Replace 2D display with 3D display
        curl_display_3d = always_redraw(create_curl_display_3d)

        # Make curl vector dynamic BEFORE camera tilt (prevents discontinuity)
        def get_dynamic_curl():
            pos_3d = axes.p2c(probe_dot.get_center())
            curl_mag_current = curl_magnitude(pos_3d)  # Keep sign for direction
            length = np.clip(curl_mag_current * 5, -3.0, 3.0)  # Very sensitive scaling

            # Use 3D arrow with proper z-direction in axes coordinates
            # For negative curl, swap start and end to fix arrow head position
            z_dir = axes.c2p(0, 0, abs(length)) - axes.c2p(0, 0, 0)
            if length >= 0:
                return Arrow3D(
                    start=probe_dot.get_center(),
                    end=probe_dot.get_center() + z_dir,
                    color=YELLOW,
                    thickness=0.02,
                    height=0.3,
                    base_radius=0.08
                )
            else:
                # For negative curl, flip the arrow direction
                return Arrow3D(
                    start=probe_dot.get_center(),
                    end=probe_dot.get_center() - z_dir,
                    color=YELLOW,
                    thickness=0.02,
                    height=0.3,
                    base_radius=0.08
                )

        curl_vector.add_updater(lambda m: m.become(get_dynamic_curl()))

        # Seamlessly tilt camera to reveal 3D while curl vector updates smoothly
        # Narration: "The larger the magnitude of this vector..."
        # Transition from 2D display to 3D display
        self.add_fixed_in_frame_mobjects(curl_display_3d)
        self.play(
            FadeOut(curl_display_2d),
            FadeIn(curl_display_3d),
            run_time=0.5
        )

        self.move_camera(phi=65*DEGREES, theta=-90*DEGREES, run_time=2.5)
        self.wait(0.5)

        # Move dot: strong curl → weak curl → strong curl (avoiding negative regions)
        # Narration: "the stronger the local rotation"
        # Path: strong vortex → midpoint between vortices → medium vortex
        mid_point = np.array([0.5, 0.0, 0.0])  # Midpoint between vortices (small curl)
        end_point = vortex_centers[1]  # Medium vortex (large curl)

        # Create a smooth path through 3 points
        def smooth_path(t):
            if t < 0.5:
                # First half: strong vortex to midpoint
                s = t * 2  # Scale to [0, 1]
                x = probe_center[0] + s * (mid_point[0] - probe_center[0])
                y = probe_center[1] + s * (mid_point[1] - probe_center[1])
            else:
                # Second half: midpoint to medium vortex
                s = (t - 0.5) * 2  # Scale to [0, 1]
                x = mid_point[0] + s * (end_point[0] - mid_point[0])
                y = mid_point[1] + s * (end_point[1] - mid_point[1])
            return axes.c2p(x, y, 0)

        move_path = ParametricFunction(
            smooth_path,
            t_range=[0, 1]
        )

        # Add slow camera rotation while dot moves
        self.begin_ambient_camera_rotation(rate=0.15)

        self.play(
            MoveAlongPath(probe_dot, move_path, run_time=4, rate_func=smooth)
        )

        # Continue rotating briefly
        self.wait(1.5)

        # Stop camera rotation
        self.stop_ambient_camera_rotation()

        # Clean up updaters
        sampling_circle.clear_updaters()
        curl_vector.clear_updaters()

        # Final hold
        self.wait(1)
