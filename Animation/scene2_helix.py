from manim import *
import numpy as np

# ==================================================
# SCENE 2 — HELICAL PATH: ROTATION + FORWARD DRIFT
# Narration: ~18 seconds
# Total runtime: ~20 seconds
# ==================================================

class Scene2_HelicalPath(ThreeDScene):
    def construct(self):
        # Camera setup - oblique angle to show both rotation and forward motion
        self.set_camera_orientation(phi=65*DEGREES, theta=45*DEGREES, zoom=0.8)

        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-1, 8, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=9,
            y_length=6,
            z_length=6
        )

        # Make axis arrow tips smaller
        for axis in axes:
            if axis.tip:
                axis.tip.scale(0.5)

        # Define helical vector field: rotation in y-z plane + forward motion along x
        # Adjusted for 2 complete turns in 10 seconds
        def helical_field(point):
            x, y, z = point
            # Rotational component in y-z plane (around x-axis)
            # Increased rotation rate for 2 turns
            v_rot_y = -z * 1.25
            v_rot_z = y * 1.25
            # Forward drift along x-axis
            v_forward_x = 1.0
            return np.array([v_forward_x, v_rot_y, v_rot_z])

        # Create vector field arrows with 3b1b styling
        velocity_field = ArrowVectorField(
            helical_field,
            x_range=[0, 7, 1.0],
            y_range=[-2.5, 2.5, 0.8],
            z_range=[-2.5, 2.5, 0.8],
            length_func=lambda norm: 0.35 * sigmoid(norm),
            color=BLUE_E,
            opacity=0.6
        )

        # Animation: Create axes and vector field (0-3s)
        # Narration: "Now let's move toward helicity. Picture a 3D vector field..."
        self.play(Create(axes, run_time=1.5))
        self.wait(0.3)
        self.play(Create(velocity_field, run_time=2))
        self.wait(0.5)

        # Create single particle - start at larger radius for bigger curl
        particle = Sphere(radius=0.08, color=YELLOW)
        particle.move_to(axes.c2p(0, 1.0, 0))

        # Create trail for particle
        trail = TracedPath(particle.get_center, stroke_color=YELLOW, stroke_width=3, stroke_opacity=0.8)

        # Narration: "...and track a single fluid particle as it moves."
        # Fade out most arrows, focus on particle
        self.play(
            velocity_field.animate.set_opacity(0.5),
            FadeIn(particle),
            run_time=1.5
        )
        self.wait(0.3)

        # Add trail to the scene
        self.add(trail)

        # Track particle position in coordinate space
        particle_coord = [0, 1.0, 0]  # Starting coordinates

        # Define particle motion function (follows helical flow)
        def update_particle(mob, dt):
            # Apply helical velocity field in coordinate space
            velocity = helical_field(np.array(particle_coord))
            # Update coordinate position
            particle_coord[0] += velocity[0] * dt * 1.0
            particle_coord[1] += velocity[1] * dt * 1.0
            particle_coord[2] += velocity[2] * dt * 1.0
            # Move particle to new world position
            particle.move_to(axes.c2p(*particle_coord))

        particle.add_updater(update_particle)

        # Camera tracking updater - follows particle without moving the scene
        def update_camera(mob, dt):
            # Update camera to follow particle's world position
            current_phi = self.camera.get_phi()
            current_theta = self.camera.get_theta()
            self.set_camera_orientation(
                phi=current_phi,
                theta=current_theta,
                frame_center=particle.get_center()
            )

        camera_tracker = VGroup()
        camera_tracker.add_updater(update_camera)
        self.add(camera_tracker)

        # Start slow camera rotation to show the helix from different angles
        self.begin_ambient_camera_rotation(rate=0.1)

        # Let particle travel for 10 seconds
        # Narration: "In this example, the particle doesn't only rotate around an axis—
        # it also drifts forward along that axis. As a result, their trajectories
        # become helices: rotation plus translation aligned with that rotation."
        self.wait(10)

        # Stop camera rotation and tracking
        self.stop_ambient_camera_rotation()
        camera_tracker.clear_updaters()
