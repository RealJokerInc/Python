from manim import *
import numpy as np

class TrailRespawnTest(ThreeDScene):
    """
    SOLUTION 2: Dynamic Spawning with Timer
    - New particles spawn continuously every 0.2-0.5 seconds
    - Dead particles removed from scene completely
    - Each particle has its own lifecycle
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"

        # Smaller parameters for faster testing
        rotation_strength = 0.6
        drift_strength = 2.5  # Faster drift to reach end quicker

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
            x_range=[-4, 4, 2],
            y_range=[-2, 2, 1.2],
            z_range=[-2, 2, 1.2],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=YELLOW,
            opacity=0.4,
        )

        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=-40 * DEGREES,
            zoom=0.7
        )

        self.add(vector_field)

        # === SOLUTION 2: DYNAMIC SPAWNING SYSTEM ===

        # Lists to track active particles and their data
        active_particles = []  # List of dict: {'particle': Dot3D, 'trail': TracedPath, 'state': {...}}

        # Spawning parameters
        spawn_timer = 0.0
        spawn_interval = 0.3  # Spawn new particle every 0.3 seconds
        max_particles = 30  # Cap on total particle count

        def spawn_new_particle():
            """Create and add a new particle with fresh trail to scene"""
            # Random spawn position throughout volume
            x0 = np.random.uniform(-4, 4)
            y0 = np.random.uniform(-2, 2)
            z0 = np.random.uniform(-2, 2)
            start = np.array([x0, y0, z0])

            # Create new particle
            particle = Dot3D(
                point=start,
                radius=0.05,
                color=BLUE_B,
            )
            particle.set_opacity(0.0)  # Start invisible

            # Create new trail
            trail = TracedPath(
                particle.get_center,
                stroke_color=BLUE,
                stroke_width=2,
                stroke_opacity=[0, 0.7],
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
                    'max_age': np.random.uniform(3, 6),
                    'alive': True,
                }
            }

            active_particles.append(particle_data)

        # Spawn initial batch with staggered ages for smooth start
        for _ in range(10):
            spawn_new_particle()
            # Give initial particles random ages so they don't all die at once
            active_particles[-1]['state']['age'] = np.random.uniform(0, 3)

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

        # Animation
        self.wait(10)  # 10 seconds to see continuous spawning

        scene_manager.remove_updater(scene_updater)
        self.wait(0.5)

class test2(ThreeDScene):
    """
    Version with traced paths that show the helical trajectories clearly.
    Fixed: Uses a robust respawn system that reseeds the TracedPath to prevent bugs.
    """
    def construct(self):
        self.camera.background_color = "#1a1a1a"
        
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
        
        # --- Particle Generation ---
        num_particles = 200
        np.random.seed(123)
        
        traced_paths = VGroup()
        particles = VGroup()
        
        # Configuration for boundaries
        fade_zone = 6.5       # Start fading out
        despawn_zone = 8.0    # Hard reset point
        respawn_start_x = -7.5 # Base respawn X coordinate
        
        # Initialize particles randomly throughout the volume
        for i in range(num_particles):
            x0 = np.random.uniform(-7, 4)
            y0 = np.random.uniform(-3.5, 3.5)
            z0 = np.random.uniform(-3.5, 3.5)
            
            start = np.array([x0, y0, z0])
            
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

        # Track opacities manually to handle fading
        particle_opacities = [1.0] * num_particles

        def update_particles(mob, dt):
            for i, particle in enumerate(mob):
                pos = particle.get_center()
                vel = helical_field(pos)
                
                # Move particle
                new_pos = pos + vel * dt * 2.0
                
                # --- BOUNDARY LOGIC ---
                
                # 1. Check if particle has left the scene (Right Side)
                if new_pos[0] > despawn_zone:
                    # RESPWAN LOGIC
                    # Generate new random coordinates
                    rand_y = np.random.uniform(-3.5, 3.5)
                    rand_z = np.random.uniform(-3.5, 3.5)
                    # Add slight randomness to X so they don't all appear in a flat vertical wall
                    rand_x = respawn_start_x + np.random.uniform(-0.5, 0.5)
                    
                    new_pos = np.array([rand_x, rand_y, rand_z])
                    
                    # Reset opacity to 0 (invisible at start)
                    particle_opacities[i] = 0.0
                    
                    # *** THE FIX FOR THE TRAIL ***
                    # Clear the old trail - TracedPath will automatically start tracking from new position
                    traced_paths[i].clear_points()

                # 2. Handle Fade Out (Right Side)
                elif new_pos[0] > fade_zone:
                    fade_progress = (new_pos[0] - fade_zone) / (despawn_zone - fade_zone)
                    particle_opacities[i] = max(0, 1 - fade_progress)
                
                # 3. Handle Fade In (Left Side / Just Respawned)
                else:
                    if particle_opacities[i] < 1.0:
                        # Fade in speed
                        particle_opacities[i] = min(1.0, particle_opacities[i] + dt * 2)

                # Soft boundary for Y and Z (Bounce/Clamp to keep them in view)
                if abs(new_pos[1]) > 4:
                    new_pos[1] = np.sign(new_pos[1]) * 4
                if abs(new_pos[2]) > 4:
                    new_pos[2] = np.sign(new_pos[2]) * 4

                # Apply updates
                particle.move_to(new_pos)
                particle.set_opacity(particle_opacities[i])
        
        particles.add_updater(update_particles)
        
        # Animation Sequence
        self.wait(1.5)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(11)
        self.stop_ambient_camera_rotation()
        
        particles.remove_updater(update_particles)
        self.wait(0.5)