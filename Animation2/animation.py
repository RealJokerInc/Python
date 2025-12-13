from manim import *

class DeanDerivation(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # SCENE 1: The Geometry & Setup
        # ---------------------------------------------------------
        
        # Title
        title = Text("Dean's Equations Derivation", font_size=40).to_edge(UP)
        subtitle = Text("From the Vorticity Transport Equation", font_size=24, color=GRAY).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)

        # Draw a schematic of a curved pipe
        # We represent this conceptually with an annulus segment
        pipe_curve = Arc(radius=3, angle=PI/3, arc_center=DOWN*2+LEFT*3, color=BLUE)
        pipe_width = 0.8
        inner_wall = Arc(radius=3-pipe_width/2, angle=PI/3, arc_center=DOWN*2+LEFT*3, color=WHITE)
        outer_wall = Arc(radius=3+pipe_width/2, angle=PI/3, arc_center=DOWN*2+LEFT*3, color=WHITE)
        
        pipe_group = VGroup(pipe_curve, inner_wall, outer_wall)
        
        # Coordinate labels
        coords = MathTex(r"(r, \theta, z)").next_to(pipe_group, RIGHT)
        curvature_note = Tex(r"Curvature: $1/R$", font_size=24).next_to(coords, DOWN)

        self.play(Create(inner_wall), Create(outer_wall))
        self.play(Write(coords), FadeIn(curvature_note))
        self.wait(2)
        
        self.play(FadeOut(pipe_group), FadeOut(coords), FadeOut(curvature_note), FadeOut(subtitle))

        # ---------------------------------------------------------
        # SCENE 2: The Curl Form (Vorticity Equation)
        # ---------------------------------------------------------
        
        header = Text("Step 1: The Curl of Navier-Stokes", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Transform(title, header))

        # Standard NS
        ns_eq = MathTex(
            r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}"
        )
        self.play(Write(ns_eq))
        self.wait(1)

        # Apply Curl Operator
        curl_op = MathTex(r"\nabla \times (\dots)").set_color(RED).next_to(ns_eq, LEFT)
        self.play(FadeIn(curl_op))
        self.play(ns_eq.animate.shift(RIGHT*0.5))
        
        # Transition to Vorticity Equation
        vorticity_def = MathTex(r"\boldsymbol{\omega} = \nabla \times \mathbf{u}", color=BLUE).to_edge(DOWN)
        self.play(Write(vorticity_def))
        self.wait(1)

        vort_eq = MathTex(
            r"\frac{D \boldsymbol{\omega}}{D t} = (\boldsymbol{\omega} \cdot \nabla) \mathbf{u} + \nu \nabla^2 \boldsymbol{\omega}"
        ).scale(1.2)
        
        self.play(
            FadeOut(curl_op), 
            FadeOut(vorticity_def), 
            ReplacementTransform(ns_eq, vort_eq)
        )
        self.wait(2)

        # Move to top for next step
        self.play(vort_eq.animate.to_edge(UP).scale(0.8), FadeOut(title))

        # ---------------------------------------------------------
        # SCENE 3: Simplifying for Dean Flow
        # ---------------------------------------------------------
        
        # Assumptions List
        assumptions = VGroup(
            Text("1. Steady State", font_size=24),
            Text("2. Fully Developed (z-independent)", font_size=24),
            Tex(r"3. Small Curvature Ratio ($\delta = a/R \ll 1$)", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(vort_eq, DOWN, buff=1)
        
        self.play(Write(assumptions))
        self.wait(2)

        # Define Stream Function
        stream_func = MathTex(
            r"u = \frac{1}{r} \frac{\partial \psi}{\partial \theta}, \quad v = -\frac{\partial \psi}{\partial r}"
        ).next_to(assumptions, DOWN, buff=1)
        
        self.play(Write(stream_func))
        self.wait(2)
        
        self.play(FadeOut(assumptions), FadeOut(vort_eq))
        self.play(stream_func.animate.to_edge(UP))

        # ---------------------------------------------------------
        # SCENE 4: The Perturbation & Result
        # ---------------------------------------------------------

        # The biharmonic operator text
        nabla4 = MathTex(r"\nabla^4 \psi").set_color(RED)
        inertial = MathTex(r"+ \frac{1}{r} \left( \frac{\partial \psi}{\partial \theta} \frac{\partial (\nabla^2 \psi)}{\partial r} - \frac{\partial \psi}{\partial r} \frac{\partial (\nabla^2 \psi)}{\partial \theta} \right)")
        dean_term = MathTex(r"= -2 \text{De} \dots") # Simplified visual representation of the forcing
        
        full_deriv = VGroup(nabla4, inertial, dean_term).arrange(RIGHT)
        
        explanation = Text("Applying the curl to curved coordinates yields:", font_size=28).next_to(stream_func, DOWN)
        self.play(Write(explanation))
        
        # Write the final result (The Dean Equation for Stream Function)
        # Usually expressed as Bi-harmonic of Psi driven by axial velocity W interaction
        
        final_eq_1 = MathTex(
            r"\nu \nabla^4 \psi + \frac{1}{r} \frac{\partial(\psi, \nabla^2 \psi)}{\partial(r, \theta)}",
            r"=",
            r"\frac{2}{R} \frac{\partial w^2}{\partial y}" # Cartesian approximation for clarity or...
        ).scale(0.9)

        # Let's use the strictly polar coordinate version often seen in textbooks
        final_eq_proper = MathTex(
            r"\Delta^2 \psi - \frac{1}{r} \frac{\partial(\psi, \Delta \psi)}{\partial(r, \theta)}",
            r"=",
            r"2 \text{De} \sin\theta \frac{\partial w}{\partial r}" # Simplified coupling
        ).scale(1.0)

        # Visualizing the components
        brace_viscous = Brace(final_eq_proper[0][0:4], UP)
        text_viscous = brace_viscous.get_text("Viscous Diffusion").scale(0.6)
        
        brace_inertial = Brace(final_eq_proper[0][5:], DOWN)
        text_inertial = brace_inertial.get_text("Convective Transport").scale(0.6)
        
        brace_forcing = Brace(final_eq_proper[2], UP)
        text_forcing = brace_forcing.get_text("Centrifugal Forcing (Dean)").scale(0.6)

        self.play(FadeOut(explanation), ReplacementTransform(stream_func, final_eq_proper))
        self.wait(1)
        self.play(
            GrowFromCenter(brace_viscous), Write(text_viscous),
            GrowFromCenter(brace_inertial), Write(text_inertial),
            GrowFromCenter(brace_forcing), Write(text_forcing)
        )
        
        self.wait(3)

        # ---------------------------------------------------------
        # SCENE 5: Conclusion
        # ---------------------------------------------------------
        
        final_box = SurroundingRectangle(final_eq_proper, color=YELLOW, buff=0.2)
        label = Text("Dean's Equation (Stream Function Form)", font_size=24, color=YELLOW).next_to(final_box, DOWN)
        
        self.play(Create(final_box), Write(label))
        self.play(FadeOut(brace_viscous), FadeOut(text_viscous), FadeOut(brace_inertial), FadeOut(text_inertial), FadeOut(brace_forcing), FadeOut(text_forcing))
        
        self.wait(3)
        