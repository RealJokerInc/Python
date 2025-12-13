from manim import *
import numpy as np

class Scene4_Final(ThreeDScene):
    """
    Scene 4 - Split screen with borders and labels from Scene4_improved
    But using EXACT animation logic from Scene2 (left) and Scene3 (right)
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"

        # ===== BORDER AND LABEL SETUP from Scene4_improved =====
        # Static camera - straight on view, centered at origin
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-90 * DEGREES,
            zoom=0.55,
            frame_center=ORIGIN  # Ensure camera is centered
        )

        # Border boxes - fixed to screen
        box_width = 6.8
        box_height = 6.2

        border_left = Rectangle(
            width=box_width,
            height=box_height,
            color=BLUE_D,
            stroke_width=3,
            stroke_opacity=0.7
        )
        border_left.move_to(LEFT * 3.6)

        border_right = Rectangle(
            width=box_width,
            height=box_height,
            color=GREEN_D,
            stroke_width=3,
            stroke_opacity=0.7
        )
        border_right.move_to(RIGHT * 3.6)

        # Labels
        label_left = Text("Helical flow", font_size=28, color=WHITE)
        label_left.next_to(border_left, DOWN, buff=0.25)

        label_right = Text("Pure rotational flow", font_size=28, color=WHITE)
        label_right.next_to(border_right, DOWN, buff=0.25)

        # Add borders and labels as fixed frame objects
        self.add_fixed_in_frame_mobjects(border_left, border_right, label_left, label_right)

        # ===== WORLD SHIFTS =====
        # Align the 3D content to be centered within the border boxes
        # The borders are at LEFT * 3.6 and RIGHT * 3.6
        # We want the 3D world origins to align with those positions
        left_shift = LEFT * 3.6
        right_shift = RIGHT * 3.6

        # ===== LEFT SIDE: Scene 2 (Helical Flow) - EXACT COPY =====
        # Parameters from Scene2
        rotation_strength_left = 0.6
        drift_strength_left = 1.2

        def helical_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            v_y = -z * rotation_strength_left / (1 + r * 0.15)
            v_z = y * rotation_strength_left / (1 + r * 0.15)
            v_x = drift_strength_left
            return np.array([v_x, v_y, v_z])

        # Vector field from Scene2
        vector_field_left = ArrowVectorField(
            helical_field,
            x_range=[-6, 6, 2],
            y_range=[-3, 3, 1.2],
            z_range=[-3, 3, 1.2],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=YELLOW,
            opacity=0.55,
        ).shift(left_shift)

        # ===== RIGHT SIDE: Scene 3 (Pure Rotation) - EXACT COPY =====
        rotation_strength_right = 1.2
        drift_strength_right = 0.0

        def pure_rotation_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            v_y = -z * rotation_strength_right / (1 + r * 0.15)
            v_z = y * rotation_strength_right / (1 + r * 0.15)
            v_x = drift_strength_right
            return np.array([v_x, v_y, v_z])

        # Vector field from Scene3
        vector_field_right = ArrowVectorField(
            pure_rotation_field,
            x_range=[-6, 6, 2],
            y_range=[-3, 3, 1.2],
            z_range=[-3, 3, 1.2],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=YELLOW,
            opacity=0.55,
        ).shift(right_shift)

        # Create both vector fields
        self.play(
            Create(vector_field_left),
            Create(vector_field_right),
            Write(label_left),
            Write(label_right),
            run_time=2
        )
        self.wait(0.3)

        # ===== LEFT PARTICLES (from Scene2) =====
        num_particles = 200
        np.random.seed(123)

        traced_paths_left = VGroup()
        particles_left = VGroup()

        for i in range(num_particles):
            x0 = np.random.uniform(-7, 4)
            y0 = np.random.uniform(-3.5, 3.5)
            z0 = np.random.uniform(-3.5, 3.5)
            start = np.array([x0, y0, z0]) + left_shift

            particle = Dot3D(
                point=start,
                radius=0.035,
                color=BLUE_B,
            )
            particles_left.add(particle)

            path = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.2,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.8,
            )
            traced_paths_left.add(path)

        # LEFT particle state tracking - lifetime-based system
        particle_opacities_left = [1.0 for _ in range(num_particles)]
        particle_lifetimes_left = [np.random.uniform(0, 8) for _ in range(num_particles)]
        max_lifetime = 8.0
        fade_duration = 1.0

        def update_particles_left(mob, dt=0):
            for i, particle in enumerate(mob):
                pos = particle.get_center() - left_shift  # Transform to local coords
                vel = helical_field(pos)
                new_pos = pos + vel * dt * 2.0

                # Update lifetime
                particle_lifetimes_left[i] += dt

                # Check if particle should respawn
                if particle_lifetimes_left[i] > max_lifetime or abs(new_pos[0]) > 8 or abs(new_pos[1]) > 4 or abs(new_pos[2]) > 4:
                    # Respawn at random location
                    new_pos[0] = np.random.uniform(-7, 4)
                    new_pos[1] = np.random.uniform(-3.5, 3.5)
                    new_pos[2] = np.random.uniform(-3.5, 3.5)
                    particle_lifetimes_left[i] = 0.0
                    particle_opacities_left[i] = 0.0

                    # Clear path and let it restart
                    traced_paths_left[i].clear_points()

                # Handle opacity based on lifetime
                if particle_lifetimes_left[i] < fade_duration:
                    particle_opacities_left[i] = particle_lifetimes_left[i] / fade_duration
                elif particle_lifetimes_left[i] > (max_lifetime - fade_duration):
                    remaining = max_lifetime - particle_lifetimes_left[i]
                    particle_opacities_left[i] = remaining / fade_duration
                else:
                    particle_opacities_left[i] = 1.0

                particle.move_to(new_pos + left_shift)  # Transform back to world coords
                particle.set_opacity(particle_opacities_left[i])

        # ===== RIGHT PARTICLES (from Scene3) =====
        traced_paths_right = VGroup()
        particles_right = VGroup()

        for i in range(num_particles):
            x0 = np.random.uniform(-5, 5)
            y0 = np.random.uniform(-3.5, 3.5)
            z0 = np.random.uniform(-3.5, 3.5)
            start = np.array([x0, y0, z0]) + right_shift

            particle = Dot3D(
                point=start,
                radius=0.035,
                color=BLUE_B,
            )
            particles_right.add(particle)

            path = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.2,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.8,
            )
            traced_paths_right.add(path)

        def update_particles_right(mob, dt=0):
            for i, particle in enumerate(mob):
                pos = particle.get_center() - right_shift  # Transform to local coords
                vel = pure_rotation_field(pos)
                new_pos = pos + vel * dt * 2.0

                # Scene3 boundary logic (no respawn)
                if abs(new_pos[1]) > 4:
                    new_pos[1] = np.sign(new_pos[1]) * 4
                if abs(new_pos[2]) > 4:
                    new_pos[2] = np.sign(new_pos[2]) * 4

                particle.move_to(new_pos + right_shift)  # Transform back to world coords

        # Add particles and traced paths
        self.add(traced_paths_left, particles_left, traced_paths_right, particles_right)

        # Add updaters
        particles_left.add_updater(update_particles_left)
        particles_right.add_updater(update_particles_right)

        # Animate - STATIC camera (no rotation)
        self.wait(1.5)
        # NO camera rotation per user's request for Scene4
        self.wait(11)

        # Clean up
        particles_left.remove_updater(update_particles_left)
        particles_right.remove_updater(update_particles_right)
        self.wait(0.5)
