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


class HelicityWithTracedPaths(ThreeDScene):
    """
    Version with traced paths that show the helical trajectories clearly
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"
        
        rotation_strength = 1.2
        drift_strength = 0.5
        helix_turns = 4  # Number of rotations globally
        
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
        
        # Generate many particles with traced paths
        num_particles = 200
        np.random.seed(123)
        
        # Create traced paths
        traced_paths = VGroup()
        particles = VGroup()
        
        for i in range(num_particles):
            # Random starting position
            x0 = np.random.uniform(-7, 4)
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
            
            # Create traced path for this particle
            path = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=1.2,
                stroke_opacity=[0, 0.6],  # Fade effect
                dissipating_time=0.8,  # Trail dissipates
            )
            traced_paths.add(path)
        
        self.add(traced_paths, particles)
        
        # Animate particles along the vector field
        dt = 0.02
        total_frames = int(12 / dt)
        
        # Store velocities for smooth animation
        velocities = [helical_field(p.get_center()) for p in particles]
        
        def update_particles(mob, delta_t):
            for i, particle in enumerate(mob):
                pos = particle.get_center()
                vel = helical_field(pos)
                new_pos = pos + vel * delta_t * 2.0
                
                # Wrap x
                if new_pos[0] > 8:
                    new_pos[0] = -8
                elif new_pos[0] < -8:
                    new_pos[0] = 8
                
                # Soft boundary for y, z
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


# To render, use one of these commands:
# manim -pql helicity_visualization.py HelicityVisualization
# manim -pql helicity_visualization.py HelicityVisualizationOptimized  
# manim -pql helicity_visualization.py HelicityWithTracedPaths
#
# For higher quality:
# manim -pqm helicity_visualization.py HelicityWithTracedPaths
#
# Flags: -p (preview), -ql (low quality), -qm (medium), -qh (high)