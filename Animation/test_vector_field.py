from manim import *
import numpy as np

class VectorFieldTest(ThreeDScene):
    def construct(self):
        # Set up the 3D camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Title
        title = Text("Vector Field: Vorticity & Helicity", font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()

        # Define a vector field with interesting vorticity
        # Using a helical/swirling field: v = (-y, x, z)
        def velocity_field(pos):
            x, y, z = pos
            return np.array([-y, x, 0.5 * z])

        # Define vorticity (curl of velocity field)
        # For v = (-y, x, 0.5*z), curl(v) = (0, 0, 2)
        def vorticity_field(pos):
            x, y, z = pos
            # ∇ × v = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)
            # For our field: (0 - 0, 0 - 0, 1 - (-1)) = (0, 0, 2)
            return np.array([0.0, 0.0, 2.0])

        # Define helicity density (v · ω where ω = ∇ × v)
        def helicity_density(pos):
            v = velocity_field(pos)
            omega = vorticity_field(pos)
            return np.dot(v, omega)

        # Create coordinate axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 1],
            x_length=6,
            y_length=6,
            z_length=4
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="z")

        self.play(Create(axes), Write(axes_labels))
        self.wait()

        # Create velocity vector field
        velocity_vectors = ArrowVectorField(
            velocity_field,
            x_range=[-2, 2, 0.8],
            y_range=[-2, 2, 0.8],
            z_range=[-1, 1, 0.8],
            length_func=lambda norm: 0.4 * sigmoid(norm),
            color=BLUE
        )

        # Label for velocity field
        v_label = MathTex(r"\vec{v} = (-y, x, 0.5z)", font_size=30)
        v_label.to_edge(LEFT).shift(UP * 2)
        self.add_fixed_in_frame_mobjects(v_label)

        self.play(Create(velocity_vectors), Write(v_label))
        self.wait()

        # Rotate camera to show the field
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # Show vorticity formula
        vorticity_formula = MathTex(
            r"\vec{\omega} = \nabla \times \vec{v} = (0, 0, 2)",
            font_size=28
        )
        vorticity_formula.to_edge(LEFT).shift(UP * 1)
        self.add_fixed_in_frame_mobjects(vorticity_formula)
        self.play(Write(vorticity_formula))
        self.wait()

        # Create vorticity vectors (constant in z-direction)
        vorticity_vectors = ArrowVectorField(
            vorticity_field,
            x_range=[-2, 2, 1.2],
            y_range=[-2, 2, 1.2],
            z_range=[-1, 1, 1.2],
            length_func=lambda norm: 0.3 * norm / norm if norm != 0 else 0.3,
            color=RED
        )

        self.play(
            velocity_vectors.animate.set_opacity(0.3),
            Create(vorticity_vectors)
        )
        self.wait(2)

        # Show helicity formula
        helicity_formula = MathTex(
            r"H = \vec{v} \cdot \vec{\omega} = z",
            font_size=28
        )
        helicity_formula.to_edge(LEFT)
        self.add_fixed_in_frame_mobjects(helicity_formula)
        self.play(Write(helicity_formula))
        self.wait()

        # Create sample points colored by helicity density
        points = []

        for x in np.linspace(-2, 2, 6):
            for y in np.linspace(-2, 2, 6):
                for z in np.linspace(-1.5, 1.5, 4):
                    pos = np.array([x, y, z])
                    h = helicity_density(pos)
                    # Color based on helicity value
                    if h > 0:
                        color = interpolate_color(WHITE, GREEN, min(abs(h) / 3, 1))
                    else:
                        color = interpolate_color(WHITE, PURPLE, min(abs(h) / 3, 1))
                    points.append(Dot3D(point=pos, radius=0.05, color=color))

        helicity_dots = VGroup(*points)

        self.play(
            vorticity_vectors.animate.set_opacity(0.3),
            Create(helicity_dots)
        )

        # Final camera rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        # Fade out
        self.play(
            FadeOut(velocity_vectors),
            FadeOut(vorticity_vectors),
            FadeOut(helicity_dots),
            FadeOut(axes),
            FadeOut(axes_labels)
        )

        # Final summary with derivative forms
        summary = VGroup(
            MathTex(r"\text{Vorticity (Curl): } \vec{\omega} = \nabla \times \vec{v}", font_size=28),
            MathTex(r"= \left(\frac{\partial v_z}{\partial y} - \frac{\partial v_y}{\partial z}, " +
                    r"\frac{\partial v_x}{\partial z} - \frac{\partial v_z}{\partial x}, " +
                    r"\frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y}\right)", font_size=24),
            MathTex(r"\text{Helicity: } H = \vec{v} \cdot (\nabla \times \vec{v})", font_size=28)
        ).arrange(DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(summary)
        self.play(Write(summary))
        self.wait(2)
