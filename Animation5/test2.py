from manim import *
import numpy as np

class HelicityIntuition(ThreeDScene):
    def construct(self):
        # Define standard colors for consistency
        C_BULK = BLUE
        C_CURL = YELLOW
        C_HELI = RED

        ########################################################################
        # Scene 1: The Equation
        ########################################################################
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        # Define terms
        h_text = MathTex(r"h", color=C_HELI)
        equals = MathTex(r"=")
        u_text = MathTex(r"\mathbf{u}", color=C_BULK)
        dot_text = MathTex(r"\cdot")
        curl_text = MathTex(r"(\nabla \times \mathbf{u})", color=C_CURL)

        equation = VGroup(h_text, equals, u_text, dot_text, curl_text).arrange(RIGHT)
        equation.scale(1.5)

        title = Text("Helicity Density Intuition").to_edge(UP)

        self.play(Write(title))
        self.play(Write(equation))
        self.wait(1)

        # Labels
        u_label = Text("Bulk Flow Direction", color=C_BULK, font_size=24).next_to(u_text, DOWN, buff=1)
        curl_label = Text("Axis of Local Rotation", color=C_CURL, font_size=24).next_to(curl_text, DOWN, buff=1)
        dot_label = Text("Measure of Alignment", font_size=20).next_to(dot_text, UP, buff=0.5)

        self.play(
            TransformFromCopy(u_text, u_label),
            TransformFromCopy(curl_text, curl_label)
        )
        self.wait(1)
        self.play(Write(dot_label))
        self.wait(2)

        # Clean up labels for next scene, keep equation
        self.play(
            FadeOut(u_label), FadeOut(curl_label), FadeOut(dot_label), FadeOut(title),
            equation.animate.to_corner(UL).scale(0.7)
        )


        ########################################################################
        # Scene 2: Visualizing the Curl (The Cross-Section)
        ########################################################################
        
        # 1. Establish the plane and 2D rotation
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3], x_length=6, y_length=6, z_length=6)
        plane = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], x_length=6, y_length=6)
        plane.set_opacity(0.3)

        self.play(Create(axes), Create(plane))

        # Define a simple 2D rotational field (vortex) in the xy plane
        def vortex_field_func(pos):
            # Counter-clockwise rotation
            return np.array([-pos[1], pos[0], 0])

        # Use particles first to show the motion intuitively
        stream_lines = StreamLines(
            vortex_field_func,
            x_range=[-2, 2], y_range=[-2, 2], z_range=[0, 0],
            stroke_width=2, color=GREY, max_anchors_per_line=10
        )
        
        self.add(stream_lines)
        stream_lines.start_animation(warmup=1, flow_speed=1.5)
        self.wait(3)

        # 2. Focus on a local area (the circle boundary idea)
        # Stop particles, show representative vectors on a circle
        self.play(stream_lines.end_animation())
        self.remove(stream_lines)

        ref_circle = Circle(radius=1.5, color=WHITE, stroke_opacity=0.5)
        
        # Create vectors along the circle boundary to emphasize rotation direction
        boundary_vectors = VGroup()
        num_vecs = 8
        for i in range(num_vecs):
            angle = i * (TAU / num_vecs)
            pos = np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle), 0])
            # Vector tangent to circle
            b_vec = Arrow(start=pos, end=pos + 0.5*np.array([-np.sin(angle), np.cos(angle), 0]), color=GREY_A, buff=0)
            boundary_vectors.add(b_vec)

        self.play(Create(ref_circle), Create(boundary_vectors))
        self.wait(1)

        # 3. Reveal the Curl Vector
        # The curl of (-y, x, 0) is (0, 0, 2). It points straight up (OUT).
        curl_vector = Arrow(start=ORIGIN, end=OUT * 2.5, color=C_CURL, buff=0, stroke_width=6)
        curl_label_3d = MathTex(r"\nabla \times \mathbf{u}", color=C_CURL).next_to(curl_vector.get_end(), RIGHT + UP)
        
        # Tilt camera to see 3D structure
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES, run_time=3)
        
        self.play(GrowArrow(curl_vector), Write(curl_label_3d))
        
        # Explain that the yellow arrow represents the spinning motion of the grey arrows
        self.play(Indicate(boundary_vectors), Indicate(curl_vector))
        self.wait(2)
        
        # Clean up local rotation visualizers to focus on the main vectors
        self.play(FadeOut(ref_circle), FadeOut(boundary_vectors))


        ########################################################################
        # Scene 3: Introducing Bulk Flow and the Dot Product
        ########################################################################

        # Create the Bulk Flow vector u
        bulk_vector = Arrow(start=ORIGIN, end=OUT * 2.5, color=C_BULK, buff=0, stroke_width=6)
        u_label_3d = MathTex(r"\mathbf{u}", color=C_BULK).next_to(bulk_vector.get_end(), LEFT + UP)

        self.play(GrowArrow(bulk_vector), Write(u_label_3d))

        # Create alignment text
        alignment_text = Text("Alignment:", font_size=24).to_corner(DL).shift(UP*1.5)
        status_text = Text("Positive Helicity (Corkscrew)", color=C_HELI, font_size=24).next_to(alignment_text, DOWN)
        self.add_fixed_in_frame_mobjects(alignment_text, status_text)

        # Case 1: Aligned (already in position)
        self.wait(2)

        # Case 2: Perpendicular (Zero Helicity)
        # Rotate u to be in the xy plane
        target_u_perp = RIGHT * 2.5
        
        self.play(
            Rotate(bulk_vector, angle=PI/2, axis=RIGHT, about_point=ORIGIN),
            MaintainPositionRelativeTo(u_label_3d, bulk_vector),
            Transform(status_text, Text("Zero Helicity (Spinning Top)", color=GREY, font_size=24).next_to(alignment_text, DOWN))
        )
        # Emphasize the 90 degree angle
        right_angle = RightAngle(bulk_vector, curl_vector, length=0.4, quadrant=(-1,1))
        self.play(Create(right_angle))
        self.wait(2)
        self.play(FadeOut(right_angle))

        # Case 3: Anti-aligned (Negative Helicity)
        # Rotate u to point down
        self.play(
            Rotate(bulk_vector, angle=PI/2, axis=RIGHT, about_point=ORIGIN),
            MaintainPositionRelativeTo(u_label_3d, bulk_vector),
            Transform(status_text, Text("Negative Helicity (Reverse Corkscrew)", color=C_HELI, font_size=24).next_to(alignment_text, DOWN))
        )
        self.wait(3)


        ########################################################################
        # Scene 4: Putting it all together (A real helical flow)
        ########################################################################
        
        self.play(
            FadeOut(axes), FadeOut(plane), FadeOut(bulk_vector), FadeOut(curl_vector),
            FadeOut(u_label_3d), FadeOut(curl_label_3d), FadeOut(status_text), FadeOut(alignment_text)
        )
        self.move_camera(phi=75 * DEGREES, theta=-30 * DEGREES)

        # Define a helical field: rotates in xy, moves in z
        # Field: F = (-y, x, 1)
        # Curl of F = (0, 0, 2)
        # u dot curl u = (-y, x, 1) dot (0, 0, 2) = 2 (Positive constant helicity)

        def helix_field_func(pos):
            return np.array([-pos[1], pos[0], 1])

        # Visualize the flow with particles moving along helices
        helix_lines = StreamLines(
            helix_field_func,
            x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2],
            stroke_width=2, color=BLUE_A, max_anchors_per_line=20,
            dt=0.1, virtual_time=4
        )
        
        final_text = Text("Helicity = Moving forward while turning", font_size=30).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(final_text)

        self.add(helix_lines)
        helix_lines.start_animation(warmup=0, flow_speed=1, time_width=0.5)
        
        # Slowly rotate camera around the helix
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(8)
        
        