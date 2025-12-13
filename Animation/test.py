from manim import *
import numpy as np

class HelicityVisualization(ThreeDScene):
    def construct(self):
        # Configuration
        self.camera.background_color = "#1a1a1a"
        
        # Parameters for the helical vector field
        rotation_strength = 1.2  # How much rotation around x-axis
        drift_strength = 0.5     # Forward drift along x-axis
        
        # Define the helical vector field function
        # Vectors rotate around x-axis (in y-z plane) and drift forward along x
        def helical_field(point):
            x, y, z = point
            # Rotation component (around x-axis, in y-z plane)
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01  # Avoid division by zero
            
            # Tangential velocity (rotation around x-axis)
            v_y = -z * rotation_strength / (1 + r * 0.3)
            v_z = y * rotation_strength / (1 + r * 0.3)
            
            # Forward drift along x-axis
            v_x = drift_strength
            
            return np.array([v_x, v_y, v_z])
        
        # Create the 3D vector field with yellow arrows
        vector_field = ArrowVectorField(
            helical_field,
            x_range=[-7, 7, 2],
            y_range=[-3.5, 3.5, 1.4],
            z_range=[-3.5, 3.5, 1.4],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=YELLOW,
            opacity=0.6,
        )
        
        # Set initial camera position
        self.set_camera_orientation(
            phi=70 * DEGREES,    # Angle from z-axis
            theta=-45 * DEGREES,  # Rotation around z-axis
            zoom=0.65
        )
        
        # Add the vector field
        self.play(Create(vector_field), run_time=2)
        self.wait(0.5)
        
        # Create flow particles
        num_particles = 250
        np.random.seed(42)  # For reproducibility
        
        # Random starting positions throughout the field
        start_positions = []
        for _ in range(num_particles):
            x = np.random.uniform(-7, 5)  # Start more toward left so they flow right
            y = np.random.uniform(-3.5, 3.5)
            z = np.random.uniform(-3.5, 3.5)
            start_positions.append(np.array([x, y, z]))
        
        # Create particle dots
        particles = VGroup()
        trails = VGroup()
        trail_points = [[] for _ in range(num_particles)]
        
        for pos in start_positions:
            particle = Dot3D(
                point=pos,
                radius=0.04,
                color=BLUE,
            )
            particles.add(particle)
        
        self.add(particles)
        
        # Animation parameters
        total_time = 12  # seconds for particle flow
        dt = 0.05  # time step for simulation
        steps = int(total_time / dt)
        trail_max_length = 40  # Maximum number of points in trail
        
        # Precompute all particle positions
        all_positions = [[pos.copy() for pos in start_positions]]
        current_positions = [pos.copy() for pos in start_positions]
        
        for step in range(steps):
            new_positions = []
            for i, pos in enumerate(current_positions):
                # Get velocity from field
                vel = helical_field(pos)
                # Update position (Euler integration)
                new_pos = pos + vel * dt * 1.5  # Scale for visual speed
                
                # Boundary handling - wrap or clamp
                if new_pos[0] > 8:
                    new_pos[0] = -8
                if new_pos[0] < -8:
                    new_pos[0] = 8
                    
                # Clamp y and z
                new_pos[1] = np.clip(new_pos[1], -4, 4)
                new_pos[2] = np.clip(new_pos[2], -4, 4)
                
                new_positions.append(new_pos)
            
            current_positions = new_positions
            all_positions.append([pos.copy() for pos in current_positions])
        
        # Create trail line objects for each particle
        trail_lines = VGroup()
        for i in range(num_particles):
            trail = VMobject()
            trail.set_stroke(color=BLUE, width=1.5, opacity=0.6)
            trail_lines.add(trail)
        
        self.add(trail_lines)
        
        # Custom updater for smooth animation with trails
        frame_count = [0]
        positions_per_frame = steps // (total_time * 15)  # Assuming ~15 fps for smooth playback
        if positions_per_frame < 1:
            positions_per_frame = 1
        
        def update_particles(mob, dt):
            nonlocal trail_points
            
            frame_count[0] += 1
            step_idx = min(int(frame_count[0] * positions_per_frame), len(all_positions) - 1)
            
            # Update particle positions
            for i, particle in enumerate(particles):
                new_pos = all_positions[step_idx][i]
                particle.move_to(new_pos)
                
                # Update trail points
                trail_points[i].append(new_pos.copy())
                
                # Limit trail length (shrinking effect)
                if len(trail_points[i]) > trail_max_length:
                    trail_points[i] = trail_points[i][-trail_max_length:]
                
                # Update trail line
                if len(trail_points[i]) >= 2:
                    trail_lines[i].set_points_smoothly(trail_points[i])
                    
                    # Fading opacity effect - older parts more transparent
                    n_points = len(trail_points[i])
                    if n_points > 2:
                        # Create gradient opacity effect
                        trail_lines[i].set_stroke(
                            color=BLUE,
                            width=1.5,
                            opacity=0.5
                        )
        
        particles.add_updater(update_particles)
        
        # Camera rotation around z-axis (start after particles begin)
        self.wait(1)  # Let particles start moving
        
        # Rotate camera while particles flow
        self.begin_ambient_camera_rotation(rate=0.15)  # Slow rotation
        self.wait(total_time - 1)
        self.stop_ambient_camera_rotation()
        
        # Clean up
        particles.remove_updater(update_particles)
        
        # Final hold
        self.wait(1)


class HelicityVisualizationOptimized(ThreeDScene):
    """
    Optimized version using StreamLines for better performance
    """
    def construct(self):
        # Configuration
        self.camera.background_color = "#1a1a1a"
        
        # Parameters for the helical vector field
        rotation_strength = 1.0
        drift_strength = 0.6
        
        def helical_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            
            # Rotation around x-axis
            v_y = -z * rotation_strength / (1 + r * 0.2)
            v_z = y * rotation_strength / (1 + r * 0.2)
            
            # Forward drift
            v_x = drift_strength
            
            return np.array([v_x, v_y, v_z])
        
        # Create vector field
        vector_field = ArrowVectorField(
            helical_field,
            x_range=[-7, 7, 2.5],
            y_range=[-3, 3, 1.5],
            z_range=[-3, 3, 1.5],
            length_func=lambda norm: 0.35 * sigmoid(norm),
            color=YELLOW,
            opacity=0.5,
        )
        
        # Set camera
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-30 * DEGREES,
            zoom=0.6
        )
        
        # Show vector field
        self.play(Create(vector_field), run_time=2)
        self.wait(0.5)
        
        # Create stream lines (3b1b flow style)
        stream_lines = StreamLines(
            helical_field,
            x_range=[-8, 8, 0.5],
            y_range=[-4, 4, 0.5],
            z_range=[-4, 4, 0.5],
            stroke_width=2,
            max_anchors_per_line=30,
            padding=1,
            stroke_color=BLUE,
            stroke_opacity=0.7,
        )
        
        # Animate stream lines
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1.5)
        
        # Wait a moment then start camera rotation
        self.wait(1.5)
        
        # Rotate camera
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(11)
        self.stop_ambient_camera_rotation()
        
        # End
        self.wait(1)
        stream_lines.end_animation()


class Scene2_HelicalPath(ThreeDScene):
    """
    Version with traced paths that show the helical trajectories clearly
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"
        
        rotation_strength = 0.6  # Reduced for less rotation
        drift_strength = 1.2  # Increased for more forward motion
        helix_turns = 1.5  # Target: 1.5 rotations before leaving
        
        def helical_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            
            v_y = -z * rotation_strength / (1 + r * 0.15)
            v_z = y * rotation_strength / (1 + r * 0.15)
            v_x = drift_strength
            
            return np.array([v_x, v_y, v_z])
        
        # Vector field
        vector_field = ArrowVectorField(
            helical_field,
            x_range=[-6, 6, 2],
            y_range=[-3, 3, 1.2],
            z_range=[-3, 3, 1.2],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=YELLOW,
            opacity=0.55,
        )
        
        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=-40 * DEGREES,
            zoom=0.55
        )
        
        self.play(Create(vector_field), run_time=2)
        self.wait(0.3)

        # === DYNAMIC SPAWNING SYSTEM ===
        np.random.seed(123)

        # Lists to track active particles
        active_particles = []

        # Spawning parameters - INCREASED SPAWNING FREQUENCY
        spawn_timer = 0.0
        spawn_interval = 0.02  # Spawn every 0.02 seconds (extremely frequent - 2.5x faster)
        max_particles = 200  # Match Scene3's 200 particles exactly

        def spawn_new_particle():
            """Create and add a new particle with fresh trail to scene"""
            # Random spawn position throughout volume
            x0 = np.random.uniform(-7, 4)
            y0 = np.random.uniform(-3.5, 3.5)
            z0 = np.random.uniform(-3.5, 3.5)
            start = np.array([x0, y0, z0])

            # Create new particle
            particle = Dot3D(
                point=start,
                radius=0.035,
                color=BLUE_B,
            )
            particle.set_opacity(0.0)  # Start invisible

            # Create new trail
            trail = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.2,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.8,
            )

            # Add to scene
            self.add(trail, particle)

            # Track particle state
            particle_data = {
                'particle': particle,
                'trail': trail,
                'state': {
                    'age': 0.0,
                    'fade_in_duration': 0.5,
                    'fade_out_duration': 0.5,
                    'max_age': np.random.uniform(4, 7),  # Longer lifetime for Scene2
                    'alive': True,
                }
            }

            active_particles.append(particle_data)

        # Spawn initial batch with staggered ages - 50% of max to start
        for _ in range(100):  # 50% of 200 max particles
            spawn_new_particle()
            # Give initial particles random ages
            active_particles[-1]['state']['age'] = np.random.uniform(0, 4)

        # === UPDATER FUNCTION ===
        def scene_updater(mob, dt=0):
            nonlocal spawn_timer

            # 1. SPAWN NEW PARTICLES
            spawn_timer += dt
            if spawn_timer >= spawn_interval and len(active_particles) < max_particles:
                spawn_new_particle()
                spawn_timer = 0.0

            # 2. UPDATE LIVING PARTICLES AND IDENTIFY DEAD ONES
            particles_to_remove = []

            for i, pdata in enumerate(active_particles):
                particle = pdata['particle']
                trail = pdata['trail']
                state = pdata['state']

                # Skip if already marked dead
                if not state['alive']:
                    particles_to_remove.append(i)
                    continue

                # Increment age
                state['age'] += dt

                # Check death conditions
                should_die = (state['age'] > state['max_age'])

                if not should_die:
                    # Check boundaries
                    pos = particle.get_center()
                    if abs(pos[0]) > 8 or abs(pos[1]) > 4 or abs(pos[2]) > 4:
                        should_die = True

                if should_die:
                    # Mark as dead and schedule for removal IMMEDIATELY
                    state['alive'] = False
                    particle.set_opacity(0.0)  # Hide particle immediately
                    trail.clear_points()  # Clear trail points immediately
                    trail.set_opacity(0.0)  # Hide trail immediately
                    particles_to_remove.append(i)
                    continue

                # Normal physics for living particles
                pos = particle.get_center()
                vel = helical_field(pos)
                new_pos = pos + vel * dt * 2.0

                # Calculate opacity (fade in/out)
                if state['age'] < state['fade_in_duration']:
                    opacity = state['age'] / state['fade_in_duration']
                elif state['age'] > state['max_age'] - state['fade_out_duration']:
                    remaining = state['max_age'] - state['age']
                    opacity = remaining / state['fade_out_duration']
                else:
                    opacity = 1.0

                # Apply updates
                particle.move_to(new_pos)
                particle.set_opacity(opacity)

            # 3. REMOVE DEAD PARTICLES FROM SCENE (do this in same frame)
            for i in reversed(particles_to_remove):  # Reverse to avoid index shifting
                pdata = active_particles[i]
                self.remove(pdata['particle'], pdata['trail'])
                active_particles.pop(i)

        # Create dummy mobject to hold updater
        scene_manager = VGroup()
        scene_manager.add_updater(scene_updater)
        self.add(scene_manager)

        # Let particles start, then rotate camera
        self.wait(1.5)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(11)
        self.stop_ambient_camera_rotation()

        scene_manager.remove_updater(scene_updater)
        self.wait(0.5)


class Scene3_PureRotation(ThreeDScene):
    """
    Scene 3 - Pure rotation without forward drift (No Helicity)
    Same setup as HelicityWithTracedPaths but with drift_strength = 0
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"

        rotation_strength = 1.2
        drift_strength = 0.0  # NO FORWARD DRIFT - pure rotation only

        def pure_rotation_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01

            v_y = -z * rotation_strength / (1 + r * 0.15)
            v_z = y * rotation_strength / (1 + r * 0.15)
            v_x = drift_strength  # Zero drift

            return np.array([v_x, v_y, v_z])

        # Vector field
        vector_field = ArrowVectorField(
            pure_rotation_field,
            x_range=[-6, 6, 2],
            y_range=[-3, 3, 1.2],
            z_range=[-3, 3, 1.2],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=YELLOW,
            opacity=0.55,
        )

        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=-40 * DEGREES,
            zoom=0.55
        )

        self.play(Create(vector_field), run_time=2)
        self.wait(0.3)

        # Generate particles
        num_particles = 200
        np.random.seed(123)

        # Create traced paths
        traced_paths = VGroup()
        particles = VGroup()

        for i in range(num_particles):
            # Random starting position
            x0 = np.random.uniform(-5, 5)
            y0 = np.random.uniform(-3.5, 3.5)
            z0 = np.random.uniform(-3.5, 3.5)

            start = np.array([x0, y0, z0])

            # Create particle
            particle = Dot3D(
                point=start,
                radius=0.035,
                color=BLUE_B,
            )
            particles.add(particle)

            # Create traced path
            path = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.2,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.8,
            )
            traced_paths.add(path)

        self.add(traced_paths, particles)

        # Particle updater - no boundary wrapping needed since no drift
        def update_particles(mob, dt=0):
            for i, particle in enumerate(mob):
                pos = particle.get_center()
                vel = pure_rotation_field(pos)
                new_pos = pos + vel * dt * 2.0

                # Soft boundary for y, z (keep particles in view)
                if abs(new_pos[1]) > 4:
                    new_pos[1] = np.sign(new_pos[1]) * 4
                if abs(new_pos[2]) > 4:
                    new_pos[2] = np.sign(new_pos[2]) * 4

                particle.move_to(new_pos)

        particles.add_updater(update_particles)

        # Let particles start, then rotate camera
        self.wait(1.5)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(11)
        self.stop_ambient_camera_rotation()

        particles.remove_updater(update_particles)
        self.wait(0.5)


class Scene4_SideBySideComparison(ThreeDScene):
    """
    Scene 4 - Split screen with separate 3D worlds, axes, and borders
    LEFT: Helical flow (1.5 rotations) | RIGHT: Pure rotational flow
    Static camera angle
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"

        # Parameters - matching Scene 2 updates
        rotation_strength_helical = 0.6  # Match Scene 2
        drift_strength_helical = 1.2  # Match Scene 2
        rotation_strength_pure = 1.2  # Keep Scene 3 rotation
        num_particles_per_side = 100

        # Static camera orientation
        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=-40 * DEGREES,
            zoom=0.55
        )

        # Shifts for completely separate worlds
        left_shift = LEFT * 5.5
        right_shift = RIGHT * 5.5

        # ===== LEFT SIDE: Helical Flow =====
        # Create axes for left side
        axes_left = ThreeDAxes(
            x_range=[-4, 4, 2],
            y_range=[-3, 3, 2],
            z_range=[-3, 3, 2],
            x_length=6,
            y_length=4.5,
            z_length=4.5
        ).shift(left_shift)

        for axis in axes_left:
            if axis.tip:
                axis.tip.scale(0.4)

        # Define helical field
        def helical_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            v_y = -z * rotation_strength_helical / (1 + r * 0.15)
            v_z = y * rotation_strength_helical / (1 + r * 0.15)
            v_x = drift_strength_helical
            return np.array([v_x, v_y, v_z])

        # Vector field for left
        vector_field_left = ArrowVectorField(
            helical_field,
            x_range=[-3, 3, 1.5],
            y_range=[-2.5, 2.5, 1.2],
            z_range=[-2.5, 2.5, 1.2],
            length_func=lambda norm: 0.35 * sigmoid(norm),
            color=YELLOW,
            opacity=0.5
        ).shift(left_shift)

        # ===== RIGHT SIDE: Pure Rotation =====
        # Create axes for right side
        axes_right = ThreeDAxes(
            x_range=[-4, 4, 2],
            y_range=[-3, 3, 2],
            z_range=[-3, 3, 2],
            x_length=6,
            y_length=4.5,
            z_length=4.5
        ).shift(right_shift)

        for axis in axes_right:
            if axis.tip:
                axis.tip.scale(0.4)

        # Define pure rotation field
        def pure_rotation_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            v_y = -z * rotation_strength_pure / (1 + r * 0.15)
            v_z = y * rotation_strength_pure / (1 + r * 0.15)
            v_x = 0.0  # No drift
            return np.array([v_x, v_y, v_z])

        # Vector field for right
        vector_field_right = ArrowVectorField(
            pure_rotation_field,
            x_range=[-3, 3, 1.5],
            y_range=[-2.5, 2.5, 1.2],
            z_range=[-2.5, 2.5, 1.2],
            length_func=lambda norm: 0.35 * sigmoid(norm),
            color=YELLOW,
            opacity=0.5
        ).shift(right_shift)

        # Add borders (visual separation)
        # Note: 3D borders are complex, so we'll add them as fixed frame objects
        border_left = Rectangle(width=5, height=3.5, color=BLUE_D, stroke_width=3, stroke_opacity=0.6)
        border_left.to_edge(LEFT).shift(LEFT * 0.5)
        self.add_fixed_in_frame_mobjects(border_left)

        border_right = Rectangle(width=5, height=3.5, color=GREEN_D, stroke_width=3, stroke_opacity=0.6)
        border_right.to_edge(RIGHT).shift(RIGHT * 0.5)
        self.add_fixed_in_frame_mobjects(border_right)

        # Labels below each box
        label_left = Text("Helical flow", font_size=28, color=WHITE)
        label_left.to_edge(DOWN).shift(LEFT * 3.5 + DOWN * 0.2)
        self.add_fixed_in_frame_mobjects(label_left)

        label_right = Text("Pure rotational flow", font_size=28, color=WHITE)
        label_right.to_edge(DOWN).shift(RIGHT * 3.5 + DOWN * 0.2)
        self.add_fixed_in_frame_mobjects(label_right)

        # Create scene
        self.play(
            Create(axes_left),
            Create(axes_right),
            run_time=1.5
        )
        self.play(
            Create(vector_field_left),
            Create(vector_field_right),
            Write(label_left),
            Write(label_right),
            run_time=2
        )
        self.wait(0.5)

        # Create LEFT particles
        np.random.seed(123)
        particles_left = VGroup()
        traced_paths_left = VGroup()
        particle_opacities_left = [1.0 for _ in range(num_particles_per_side)]
        fade_zone = 5.5
        respawn_zone = -6.5

        for i in range(num_particles_per_side):
            x0 = np.random.uniform(-5, 2)
            y0 = np.random.uniform(-3, 3)
            z0 = np.random.uniform(-3, 3)
            start = np.array([x0, y0, z0]) + left_shift

            particle = Dot3D(point=start, radius=0.04, color=BLUE_B)
            particles_left.add(particle)

            path = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.5,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.7
            )
            traced_paths_left.add(path)

        # Create RIGHT particles
        particles_right = VGroup()
        traced_paths_right = VGroup()

        for i in range(num_particles_per_side):
            x0 = np.random.uniform(-3, 3)
            y0 = np.random.uniform(-3, 3)
            z0 = np.random.uniform(-3, 3)
            start = np.array([x0, y0, z0]) + right_shift

            particle = Dot3D(point=start, radius=0.04, color=GREEN_B)
            particles_right.add(particle)

            path = TracedPath(
                particle.get_center,
                stroke_color=GREEN,
                stroke_width=1.5,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.7
            )
            traced_paths_right.add(path)

        self.add(traced_paths_left, particles_left, traced_paths_right, particles_right)

        # LEFT updater
        def update_particles_left(mob, dt=0):
            for i, particle in enumerate(mob):
                pos = particle.get_center() - left_shift
                vel = helical_field(pos)
                new_pos = pos + vel * dt * 2.0

                if new_pos[0] > fade_zone:
                    fade_progress = (new_pos[0] - fade_zone) / (7 - fade_zone)
                    particle_opacities_left[i] = max(0, 1 - fade_progress)

                    if new_pos[0] > 7:
                        new_pos[0] = respawn_zone
                        new_pos[1] = np.random.uniform(-3, 3)
                        new_pos[2] = np.random.uniform(-3, 3)
                        particle_opacities_left[i] = 0
                        traced_paths_left[i].clear_points()
                else:
                    if particle_opacities_left[i] < 1.0:
                        particle_opacities_left[i] = min(1.0, particle_opacities_left[i] + dt * 2)

                if abs(new_pos[1]) > 3.5:
                    new_pos[1] = np.sign(new_pos[1]) * 3.5
                if abs(new_pos[2]) > 3.5:
                    new_pos[2] = np.sign(new_pos[2]) * 3.5

                particle.move_to(new_pos + left_shift)
                particle.set_opacity(particle_opacities_left[i])

        # RIGHT updater
        def update_particles_right(mob, dt=0):
            for i, particle in enumerate(mob):
                pos = particle.get_center() - right_shift
                vel = pure_rotation_field(pos)
                new_pos = pos + vel * dt * 2.0

                if abs(new_pos[1]) > 3.5:
                    new_pos[1] = np.sign(new_pos[1]) * 3.5
                if abs(new_pos[2]) > 3.5:
                    new_pos[2] = np.sign(new_pos[2]) * 3.5

                particle.move_to(new_pos + right_shift)

        particles_left.add_updater(update_particles_left)
        particles_right.add_updater(update_particles_right)

        # Animate - STATIC camera (no rotation per user request)
        self.wait(13)

        particles_left.remove_updater(update_particles_left)
        particles_right.remove_updater(update_particles_right)
        self.wait(0.5)

class Scene4_improved(ThreeDScene):
    """
    Scene 4 - Split screen with separate 3D worlds, axes, and borders
    LEFT: Helical flow (1.5 rotations) | RIGHT: Pure rotational flow
    Static camera angle
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"

        # Parameters - matching Scene 2 updates
        rotation_strength_helical = 0.6
        drift_strength_helical = 1.2
        rotation_strength_pure = 1.2
        num_particles_per_side = 100

        # Static camera orientation - straight on view works better for split screen
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-90 * DEGREES,  # Head-on view
            zoom=0.55  # Zoomed out more to show larger scenes
        )

        # ===== CREATE 2D FRAME ELEMENTS FIRST =====
        # These stay fixed relative to the screen
        
        # Border boxes - positioned in screen space (larger)
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
        
        # Labels centered below each box
        label_left = Text("Helical flow", font_size=28, color=WHITE)
        label_left.next_to(border_left, DOWN, buff=0.25)
        
        label_right = Text("Pure rotational flow", font_size=28, color=WHITE)
        label_right.next_to(border_right, DOWN, buff=0.25)
        
        # Add as fixed frame objects
        self.add_fixed_in_frame_mobjects(border_left, border_right, label_left, label_right)

        # ===== 3D WORLD SHIFTS =====
        # Position 3D content - moved more towards center
        left_shift = LEFT * 3.6
        right_shift = RIGHT * 3.6

        # ===== LEFT SIDE: Helical Flow =====
        axes_left = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            z_range=[-2.5, 2.5, 1],
            x_length=5.5,
            y_length=4.5,
            z_length=4.5,
            axis_config={"include_tip": True, "tip_length": 0.15}
        ).shift(left_shift)

        # Define helical field (relative to local origin)
        def helical_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            v_y = -z * rotation_strength_helical / (1 + r * 0.15)
            v_z = y * rotation_strength_helical / (1 + r * 0.15)
            v_x = drift_strength_helical
            return np.array([v_x, v_y, v_z])

        # Vector field for left - larger range
        vector_field_left = ArrowVectorField(
            helical_field,
            x_range=[-2.5, 2.5, 1.2],
            y_range=[-2, 2, 1.2],
            z_range=[-2, 2, 1.2],
            length_func=lambda norm: 0.32 * sigmoid(norm),
            color=YELLOW,
            opacity=0.5
        ).shift(left_shift)

        # ===== RIGHT SIDE: Pure Rotation =====
        axes_right = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            z_range=[-2.5, 2.5, 1],
            x_length=5.5,
            y_length=4.5,
            z_length=4.5,
            axis_config={"include_tip": True, "tip_length": 0.15}
        ).shift(right_shift)

        # Define pure rotation field
        def pure_rotation_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            v_y = -z * rotation_strength_pure / (1 + r * 0.15)
            v_z = y * rotation_strength_pure / (1 + r * 0.15)
            v_x = 0.0  # No drift
            return np.array([v_x, v_y, v_z])

        # Vector field for right
        vector_field_right = ArrowVectorField(
            pure_rotation_field,
            x_range=[-2.5, 2.5, 1.2],
            y_range=[-2, 2, 1.2],
            z_range=[-2, 2, 1.2],
            length_func=lambda norm: 0.32 * sigmoid(norm),
            color=YELLOW,
            opacity=0.5
        ).shift(right_shift)

        # Create scene
        self.play(
            Create(axes_left),
            Create(axes_right),
            run_time=1.5
        )
        self.play(
            Create(vector_field_left),
            Create(vector_field_right),
            Write(label_left),
            Write(label_right),
            run_time=2
        )
        self.wait(0.5)

        # ===== PARTICLES =====
        # Bounds for particles (local coordinates)
        bounds_x = 3.0
        bounds_yz = 2.5
        fade_zone = 2.5
        respawn_x = -3.5

        # Create LEFT particles with TracedPath
        np.random.seed(123)
        particles_left = VGroup()
        traced_paths_left = VGroup()  # Use VGroup for easier indexing
        particle_opacities_left = [1.0 for _ in range(num_particles_per_side)]

        for i in range(num_particles_per_side):
            x0 = np.random.uniform(-bounds_x, bounds_x * 0.5)
            y0 = np.random.uniform(-bounds_yz, bounds_yz)
            z0 = np.random.uniform(-bounds_yz, bounds_yz)
            start = np.array([x0, y0, z0]) + left_shift

            particle = Dot3D(point=start, radius=0.04, color=BLUE_B)
            particles_left.add(particle)

            path = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.5,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.8
            )
            traced_paths_left.add(path)

        # Create RIGHT particles
        particles_right = VGroup()
        traced_paths_right = VGroup()  # Use VGroup for consistency

        for i in range(num_particles_per_side):
            x0 = np.random.uniform(-bounds_yz, bounds_yz)
            y0 = np.random.uniform(-bounds_yz, bounds_yz)
            z0 = np.random.uniform(-bounds_yz, bounds_yz)
            start = np.array([x0, y0, z0]) + right_shift

            particle = Dot3D(point=start, radius=0.04, color=GREEN_B)
            particles_right.add(particle)

            path = TracedPath(
                particle.get_center,
                stroke_color=GREEN,
                stroke_width=1.5,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.8
            )
            traced_paths_right.add(path)

        # Add all traced paths and particles to scene
        self.add(traced_paths_left, particles_left, traced_paths_right, particles_right)

        # LEFT updater - using original spawning/despawning logic
        def update_particles_left(mob, dt=0):
            for i, particle in enumerate(mob):
                pos = particle.get_center() - left_shift
                vel = helical_field(pos)
                new_pos = pos + vel * dt * 1.8

                # Fade out as particles approach exit boundary
                if new_pos[0] > fade_zone:
                    fade_progress = (new_pos[0] - fade_zone) / (bounds_x + 1.0 - fade_zone)
                    particle_opacities_left[i] = max(0, 1 - fade_progress)
                    particle.set_opacity(particle_opacities_left[i])

                    # Respawn when fully past boundary
                    if new_pos[0] > bounds_x + 1.0:
                        new_pos[0] = respawn_x
                        new_pos[1] = np.random.uniform(-bounds_yz, bounds_yz)
                        new_pos[2] = np.random.uniform(-bounds_yz, bounds_yz)
                        particle_opacities_left[i] = 0.0
                        particle.set_opacity(0.0)
                        # Clear the traced path to prevent line from old to new position
                        traced_paths_left[i].clear_points()
                else:
                    # Fade back in after respawn
                    if particle_opacities_left[i] < 1.0:
                        particle_opacities_left[i] = min(1.0, particle_opacities_left[i] + dt * 2.5)
                        particle.set_opacity(particle_opacities_left[i])

                # Boundary constraints for y and z
                if abs(new_pos[1]) > bounds_yz:
                    new_pos[1] = np.sign(new_pos[1]) * bounds_yz
                if abs(new_pos[2]) > bounds_yz:
                    new_pos[2] = np.sign(new_pos[2]) * bounds_yz

                particle.move_to(new_pos + left_shift)

        # RIGHT updater - pure rotation stays in bounds
        def update_particles_right(mob, dt=0):
            for i, particle in enumerate(mob):
                pos = particle.get_center() - right_shift
                vel = pure_rotation_field(pos)
                new_pos = pos + vel * dt * 1.8

                # Boundary constraints for all axes
                if abs(new_pos[0]) > bounds_yz:
                    new_pos[0] = np.sign(new_pos[0]) * bounds_yz
                if abs(new_pos[1]) > bounds_yz:
                    new_pos[1] = np.sign(new_pos[1]) * bounds_yz
                if abs(new_pos[2]) > bounds_yz:
                    new_pos[2] = np.sign(new_pos[2]) * bounds_yz

                particle.move_to(new_pos + right_shift)

        particles_left.add_updater(update_particles_left)
        particles_right.add_updater(update_particles_right)

        # Animate - STATIC camera (no rotation per user request)
        self.wait(13)

        particles_left.remove_updater(update_particles_left)
        particles_right.remove_updater(update_particles_right)
        self.wait(0.5)


# To render, use one of these commands:
# manim -pql helicity_visualization.py HelicityVisualization
# manim -pql helicity_visualization.py HelicityVisualizationOptimized  
# manim -pql helicity_visualization.py HelicityWithTracedPaths
#
# For higher quality:
# manim -pqm helicity_visualization.py HelicityWithTracedPaths
#

# ==================================================
# SCENE 5 — HELICITY DENSITY VISUALIZATION (REVISED)
# Sequence: Vector field → Circle + curl → Particles → Bulk flow → Equation
# ==================================================

class Scene5(ThreeDScene):
    """
    Helicity density visualization with updated sequence:
    1. Vector field appears
    2. Circle (y-z plane) + orange tangent arrows
    3. Curl vector appears
    4. Vector field fades slightly
    5. Particle flow starts (circle/arrows/curl stay visible)
    6. Bulk flow vector appears
    7. Equation appears
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"
        
        # Camera setup - 3D view
        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=-40 * DEGREES,
            zoom=0.6
        )
        
        # Color scheme
        C_BULK = BLUE
        C_CURL = YELLOW
        C_HELI = RED
        
        # ===== STEP 1: Vector field appears first =====
        rotation_strength = 0.6
        drift_strength = 1.2
        
        def helical_field(point):
            x, y, z = point
            r = np.sqrt(y**2 + z**2)
            if r < 0.01:
                r = 0.01
            
            v_y = -z * rotation_strength / (1 + r * 0.15)
            v_z = y * rotation_strength / (1 + r * 0.15)
            v_x = drift_strength
            
            return np.array([v_x, v_y, v_z])
        
        vector_field = ArrowVectorField(
            helical_field,
            x_range=[-6, 6, 2],
            y_range=[-3, 3, 1.2],
            z_range=[-3, 3, 1.2],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=YELLOW,
            opacity=0.55,
        )
        
        self.play(Create(vector_field), run_time=2)
        self.wait(0.5)
        
        # ===== STEP 2: Orange tangent arrows (no circle) =====
        circle_center = LEFT * 3

        tangent_arrows = VGroup()
        num_arrows = 8
        circle_radius = 1.2
        
        for i in range(num_arrows):
            angle = i * TAU / num_arrows
            # Position on circle (in y-z plane)
            pos_on_circle = circle_center + circle_radius * np.array([0, np.cos(angle), np.sin(angle)])
            # Tangent direction (perpendicular to radius, in y-z plane)
            tangent_dir = np.array([0, -np.sin(angle), np.cos(angle)])
            
            arrow = Arrow(
                start=pos_on_circle,
                end=pos_on_circle + tangent_dir * 0.4,
                color=ORANGE,
                buff=0,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.25
            )
            tangent_arrows.add(arrow)
        
        self.play(Create(tangent_arrows), run_time=1.5)
        self.wait(0.5)
        
        # ===== STEP 4: Curl vector appears =====
        curl_vector = Arrow(
            start=circle_center,
            end=circle_center + RIGHT * 2,
            color=C_CURL,
            buff=0,
            stroke_width=6
        )
        curl_label = MathTex(r"\nabla \times \vec{u}", color=C_CURL, font_size=32)
        curl_label.next_to(curl_vector.get_end(), UP + RIGHT * 0.5)
        self.add_fixed_in_frame_mobjects(curl_label)
        
        self.play(GrowArrow(curl_vector), Write(curl_label), run_time=1.5)
        self.wait(1)
        
        # ===== STEP 5: Fade vector field slightly =====
        self.play(vector_field.animate.set_opacity(0.25), run_time=1)
        
        # ===== STEP 6: Particle flow starts =====
        # (Circle, tangent arrows, and curl vector stay visible)
        np.random.seed(123)
        active_particles = []
        
        spawn_timer = 0.0
        spawn_interval = 0.02
        max_particles = 200
        
        def spawn_new_particle():
            x0 = np.random.uniform(-7, 4)
            y0 = np.random.uniform(-3.5, 3.5)
            z0 = np.random.uniform(-3.5, 3.5)
            start = np.array([x0, y0, z0])
            
            particle = Dot3D(
                point=start,
                radius=0.035,
                color=BLUE_B,
            )
            particle.set_opacity(0.0)
            
            trail = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.2,
                stroke_opacity=[0, 0.6],
                dissipating_time=0.8,
            )
            
            self.add(trail, particle)
            
            particle_data = {
                'particle': particle,
                'trail': trail,
                'state': {
                    'age': 0.0,
                    'fade_in_duration': 0.5,
                    'fade_out_duration': 0.5,
                    'max_age': np.random.uniform(4, 7),
                    'alive': True,
                }
            }
            
            active_particles.append(particle_data)
        
        # Spawn initial batch
        for _ in range(100):
            spawn_new_particle()
            active_particles[-1]['state']['age'] = np.random.uniform(0, 3)
        
        # Updater function
        def scene_updater(mob, dt=0):
            nonlocal spawn_timer
            
            # Spawn new particles
            spawn_timer += dt
            if spawn_timer >= spawn_interval and len(active_particles) < max_particles:
                spawn_new_particle()
                spawn_timer = 0.0
            
            # Update and remove particles
            particles_to_remove = []
            
            for i, pdata in enumerate(active_particles):
                particle = pdata['particle']
                trail = pdata['trail']
                state = pdata['state']
                
                if not state['alive']:
                    particles_to_remove.append(i)
                    continue
                
                state['age'] += dt
                
                # Death conditions
                should_die = (state['age'] > state['max_age'])
                
                if not should_die:
                    pos = particle.get_center()
                    if abs(pos[0]) > 8 or abs(pos[1]) > 4 or abs(pos[2]) > 4:
                        should_die = True
                
                if should_die:
                    state['alive'] = False
                    particle.set_opacity(0.0)
                    trail.clear_points()
                    trail.set_opacity(0.0)
                    particles_to_remove.append(i)
                    continue
                
                # Physics update
                pos = particle.get_center()
                vel = helical_field(pos)
                new_pos = pos + vel * dt * 2.0
                
                # Opacity calculation
                if state['age'] < state['fade_in_duration']:
                    opacity = state['age'] / state['fade_in_duration']
                elif state['age'] > state['max_age'] - state['fade_out_duration']:
                    remaining = state['max_age'] - state['age']
                    opacity = remaining / state['fade_out_duration']
                else:
                    opacity = 1.0
                
                particle.move_to(new_pos)
                particle.set_opacity(opacity)
            
            # Remove dead particles
            for i in reversed(particles_to_remove):
                pdata = active_particles[i]
                self.remove(pdata['particle'], pdata['trail'])
                active_particles.pop(i)
        
        scene_manager = VGroup()
        scene_manager.add_updater(scene_updater)
        self.add(scene_manager)
        
        # Let particles flow for a bit before showing bulk vector
        self.wait(3)
        
        # ===== STEP 7: Bulk flow vector appears =====
        bulk_vector = Arrow(
            start=circle_center,
            end=circle_center + RIGHT * 2,
            color=C_BULK,
            buff=0,
            stroke_width=6
        )
        bulk_vector.shift(DOWN * 0.3)  # Offset slightly so both vectors are visible
        
        bulk_label = MathTex(r"\text{bulk flow } \vec{u}", color=C_BULK, font_size=32)
        bulk_label.next_to(bulk_vector.get_end(), DOWN + RIGHT * 0.5)
        self.add_fixed_in_frame_mobjects(bulk_label)
        
        self.play(GrowArrow(bulk_vector), Write(bulk_label), run_time=1.5)
        self.wait(1)
        
        # ===== STEP 8: Equation appears =====
        equation = MathTex(
            r"h", r"=", r"\vec{u}", r"\cdot", r"(", r"\nabla \times \vec{u}", r")",
            font_size=40
        )
        equation[0].set_color(C_HELI)  # h
        equation[2].set_color(C_BULK)  # u
        equation[5].set_color(C_CURL)  # curl u
        
        equation.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(equation)
        
        self.play(Write(equation), run_time=1.5)
        
        # Continue particle flow
        self.wait(5)
        
        scene_manager.remove_updater(scene_updater)
        self.wait(0.5)
