from manim import *
import numpy as np

class DeanVortexPhysicsRevisited(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # INTRO: Dean Number Derivation
        # ---------------------------------------------------------

        # Title
        intro_title = Text("The Dean Number", font_size=48, color=YELLOW)
        self.play(Write(intro_title))
        self.wait(0.5)
        self.play(intro_title.animate.to_edge(UP).scale(0.7))

        # Step 1: Conceptual force ratio
        force_ratio = MathTex(
            r"\text{De} = \sqrt{\frac{1}{2} \cdot \frac{\text{Inertial forces} \times \text{Centrifugal forces}}{\text{Viscous forces}}}"
        ).scale(0.8)
        self.play(Write(force_ratio))
        self.wait(1)

        # Step 2: Identify forces from Navier-Stokes terms
        ns_terms = VGroup(
            MathTex(r"\text{Inertial: } (\mathbf{u} \cdot \nabla) \mathbf{u}", color=BLUE).scale(0.6),
            MathTex(r"\text{Centrifugal: } \mathbf{u}^2 / R_c", color=RED).scale(0.6),
            MathTex(r"\text{Viscous: } \nu \nabla^2 \mathbf{u}", color=GREEN).scale(0.6)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(force_ratio, DOWN, buff=0.5)

        self.play(FadeIn(ns_terms, shift=UP))
        self.wait(1)

        # Step 3: Dimensional analysis -> Dean number form
        dean_derivation = MathTex(
            r"\text{De} \sim \sqrt{\frac{U^2/D \cdot U^2/R_c}{\nu U/D^2}}"
        ).scale(0.8).move_to(force_ratio)

        self.play(
            ReplacementTransform(force_ratio, dean_derivation),
            FadeOut(ns_terms)
        )
        self.wait(0.8)

        # Step 4: Simplify to final form
        dean_simplified = MathTex(
            r"\text{De} = \frac{U D}{\nu} \sqrt{\frac{D}{2 R_c}}"
        ).scale(0.9).move_to(dean_derivation)

        self.play(TransformMatchingShapes(dean_derivation, dean_simplified))
        self.wait(0.5)

        # Step 5: Show Reynolds number substitution
        dean_final = MathTex(
            r"\text{De} = \text{Re} \sqrt{\frac{D}{2 R_c}}"
        ).scale(1.1).move_to(dean_simplified)

        re_label = MathTex(r"\text{Re} = \frac{UD}{\nu}", color=BLUE).scale(0.6).next_to(dean_final, DOWN, buff=0.5)

        self.play(
            TransformMatchingShapes(dean_simplified, dean_final),
            FadeIn(re_label, shift=UP)
        )
        self.wait(1.5)

        # Highlight the final result
        final_box = SurroundingRectangle(dean_final, color=YELLOW, buff=0.15)
        self.play(Create(final_box))
        self.wait(0.8)

        # Clear intro for main animation
        self.play(
            FadeOut(intro_title),
            FadeOut(dean_final),
            FadeOut(re_label),
            FadeOut(final_box)
        )

        # ---------------------------------------------------------
        # 1. The Setup & The Equation
        # ---------------------------------------------------------

        # Bring back the final equation from the previous derivation
        # Using the simpler representation for clarity in visualization
        equation = MathTex(
            r"\text{Viscous } \dots", r" \nabla^4 \psi", 
            r"\quad - \quad",
            r"\text{Inertial } \dots", r" J(\psi, \nabla^2 \psi)",
            r"=",
            r" \text{Centrifugal Forcing } \dots", r" 2 \text{De} \frac{\partial w}{\partial r}_{\text{approx}}"
        ).scale(0.7).to_edge(UP)
        
        # Color code terms
        equation[1].set_color(BLUE)   # Viscous
        equation[4].set_color(YELLOW) # Inertial Convection
        equation[7].set_color(RED)    # Forcing

        self.play(Write(equation))

        # Pipe Cross-section
        pipe_center = DOWN * 1.5
        radius = 2.5
        pipe = Circle(radius=radius, color=WHITE, stroke_width=4).move_to(pipe_center)
        
        # Convention: Center of curvature is to the LEFT.
        # Inner Wall = Left side (Concave). Outer Wall = Right side (Convex).
        inner_label = Text("Inner Wall\n(Concave)", font_size=16).next_to(pipe, LEFT)
        outer_label = Text("Outer Wall\n(Convex)", font_size=16).next_to(pipe, RIGHT)
        curvature_arrow = Arrow(start=LEFT*5+DOWN*1.5, end=LEFT*3+DOWN*1.5, color=GRAY)
        curvature_text = Text("To Center of\nCurvature", font_size=14, color=GRAY).next_to(curvature_arrow, UP)

        self.play(Create(pipe), FadeIn(inner_label), FadeIn(outer_label))
        #self.play(GrowArrow(curvature_arrow), Write(curvature_text))

        # ---------------------------------------------------------
        # 2. The Physics Simulation (Vector Field)
        # ---------------------------------------------------------
        
        dean_number = ValueTracker(20)
        
        # --- Physics Definition ---
        def dean_flow_field(pos):
            # Relative position to pipe center
            rel_pos = pos - pipe_center
            x, y, z = rel_pos
            r_sq = x**2 + y**2
            
            if r_sq > radius**2: 
                return np.array([0,0,0])
                
            de = dean_number.get_value()
            
            # Physics scaling
            # Intensity grows with De, but saturates slightly to prevent massive blowup
            intensity = 1.5 * np.tanh(de / 60.0) 
            
            # Center shift: Vortices move towards outer wall (Right/positive x) as De increases
            shift_x = 0.8 * np.tanh(de / 100.0)
            
            # Vortex centers (symmetric about x-axis)
            # Placed roughly at y = +/- 0.6r
            top_center_rel = np.array([shift_x, 0.6*radius, 0])
            bot_center_rel = np.array([shift_x, -0.6*radius, 0])
            
            # --- Vortex Generation ---
            # High velocity core fluid moves towards OUTER wall (Right).
            # Return flow along walls moves towards INNER wall (Left).
            # Top Vortex must be CLOCKWISE (cross radial vector with IN)
            # Bottom Vortex must be COUNTER-CLOCKWISE (cross radial vector with OUT)
            
            r_top = rel_pos - top_center_rel
            d_top = np.linalg.norm(r_top) + 0.3 # Softener to avoid infinity
            v_top = np.cross(r_top, IN) / (d_top**1.8)
            
            r_bot = rel_pos - bot_center_rel
            d_bot = np.linalg.norm(r_bot) + 0.3
            v_bot = np.cross(r_bot, OUT) / (d_bot**1.8)
            
            v_tot = (v_top + v_bot) * intensity
            
            # Boundary condition mask (no slip at walls)
            dist_from_center = np.sqrt(r_sq)
            if dist_from_center > radius - 0.3:
                mask = (radius - dist_from_center) / 0.3
                v_tot *= np.clip(mask, 0, 1)
                
            return v_tot

        # --- Vector Field Visualization ---
        # Use color to indicate magnitude, cap length to prevent blowup
        vector_field = ArrowVectorField(
            dean_flow_field,
            x_range=[pipe_center[0]-radius*0.9, pipe_center[0]+radius*0.9, 0.4],
            y_range=[pipe_center[1]-radius*0.9, pipe_center[1]+radius*0.9, 0.4],
            colors=[BLUE_E, BLUE, YELLOW, RED], # Color map for velocity magnitude
            length_func=lambda x: 0.6 * np.tanh(x) # Sigmoid length capping
        )
        
        self.play(Create(vector_field))
        self.wait(1)

        # ---------------------------------------------------------
        # 3. Connecting Equation Terms to Simulation
        # ---------------------------------------------------------

        # --- Term 3: Centrifugal Forcing ---
        brace_forcing = Brace(equation[7], DOWN, color=RED)
        text_forcing = Text("Drives fast core fluid\ntowards Outer Wall", font_size=18, color=RED).next_to(brace_forcing, DOWN, buff=0.2)
        
        # Highlight center arrows moving right
        center_flow_box = SurroundingRectangle(pipe, buff=-1.0, color=RED).stretch_to_fit_width(1.5)
        
        self.play(GrowFromCenter(brace_forcing), Write(text_forcing))
        self.play(Create(center_flow_box))
        self.wait(2)
        self.play(FadeOut(brace_forcing), FadeOut(text_forcing), FadeOut(center_flow_box))

        # --- Term 1: Viscous Diffusion ---
        brace_viscous = Brace(equation[1], DOWN, color=BLUE)
        text_viscous = brace_viscous.get_text("Resists motion,\ncreates boundary layers").scale(0.6).set_color(BLUE)
        
        # Highlight wall region
        wall_flow_ring = Annulus(inner_radius=radius-0.4, outer_radius=radius, color=BLUE).move_to(pipe_center)

        self.play(GrowFromCenter(brace_viscous), Write(text_viscous))
        self.play(Create(wall_flow_ring))
        self.wait(2)
        self.play(FadeOut(brace_viscous), FadeOut(text_viscous), FadeOut(wall_flow_ring))


        # ---------------------------------------------------------
        # 4. Animate Increasing Dean Number
        # ---------------------------------------------------------
        
        de_counter = Integer(dean_number.get_value()).set_color(RED).to_corner(DL).shift(RIGHT * 3.5)
        de_label = Text("Dean Number (De): ", font_size=24).next_to(de_counter, LEFT)
        self.play(Write(de_label), Write(de_counter))

        # Updater for vector field to redraw based on new De value
        vector_field.add_updater(lambda m: m.become(
            ArrowVectorField(
                dean_flow_field,
                x_range=[pipe_center[0]-radius*0.9, pipe_center[0]+radius*0.9, 0.4],
                y_range=[pipe_center[1]-radius*0.9, pipe_center[1]+radius*0.9, 0.4],
                colors=[BLUE_E, BLUE, YELLOW, RED],
                # Ensure length scaling stays consistent relative to new magnitudes
                length_func=lambda x: 0.6 * np.tanh(x) 
            )
        ))
        de_counter.add_updater(lambda m: m.set_value(dean_number.get_value()))

        note = Text("As De increases:\n1. Forcing dominates viscosity.\n2. Vortices intensify (color changes).\n3. Centers shift outward (Right).", 
                    font_size=18, t2c={"De": RED, "outward": RED}).to_corner(DR)
        self.play(FadeIn(note))

        # Animate De from 20 to 200
        self.play(
            dean_number.animate.set_value(200),
            run_time=8,
            rate_func=linear
        )
        
        self.wait(3)