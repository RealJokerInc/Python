from manim import *
import numpy as np

class DeanToSingleCurl(Scene):
    def construct(self):
        # ------------------------------------------------
        # 0. Title
        # ------------------------------------------------
        title = Text(
            "From Dean Counterflow to Single Curl Dominance",
            font_size=40
        ).to_edge(UP)
        self.play(Write(title))

        # ------------------------------------------------
        # 1. Cross-section of curved aorta (pipe)
        # ------------------------------------------------
        circle = Circle(radius=2, color=WHITE, stroke_width=3)
        circle_label = Text("Aortic cross-section", font_size=30).next_to(circle, DOWN)

        self.play(Create(circle), FadeIn(circle_label, shift=UP))

        # ------------------------------------------------
        # 2. Dean counter-rotating vortices (symmetric pair)
        # ------------------------------------------------
        # Helper to create a vortex arc with arrow tip
        def make_vortex(radius, center_shift, color, clockwise=True):
          """
          radius: radius of vortex arc
          center_shift: vector shift of vortex center inside the cross-section
          clockwise: direction of rotation
          """
          # For clockwise (negative angle) start at top; for ccw (positive) also start at top
          start_angle = PI/2  # start at "north"
          angle = -2 * PI if clockwise else 2 * PI

          arc = Arc(
              radius=radius,
              start_angle=start_angle,
              angle=angle,
              stroke_width=6,
              color=color,
          ).move_to(center_shift)

          # Add arrow tip in direction of circulation
          arc.add_tip(tip_length=0.2, tip_width=0.2)
          return arc

        # Two Dean vortices inside the pipe
        left_center = LEFT * 0.8
        right_center = RIGHT * 0.8

        left_vortex = make_vortex(
            radius=0.7,
            center_shift=left_center,
            color=BLUE,
            clockwise=True
        )

        right_vortex = make_vortex(
            radius=0.7,
            center_shift=right_center,
            color=PURPLE,
            clockwise=False
        )

        dean_label = Text(
            "Dean counter-rotating vortices",
            font_size=30
        ).next_to(circle, UP).shift(DOWN * 0.6)

        self.play(
            LaggedStart(
                Create(left_vortex),
                Create(right_vortex),
                lag_ratio=0.2
            ),
            FadeIn(dean_label, shift=UP),
        )
        self.wait(1)

        # ------------------------------------------------
        # 3. Introduce asymmetry (e.g. valve / upstream swirl)
        # ------------------------------------------------
        # Draw a small "valve" obstruction on upper-right side
        # Create as a simple filled rectangle positioned on the edge
        valve = Rectangle(
            width=0.6,
            height=0.4,
            color=YELLOW,
            fill_opacity=0.4,
            stroke_width=0,
        ).shift(UP * 0.3 + RIGHT * 1.5)

        valve_label = Text(
            "Asymmetry: valve / inflow swirl / wall",
            font_size=28
        ).next_to(circle, RIGHT).shift(UP * 0.5)

        self.play(
            FadeIn(valve),
            FadeIn(valve_label, shift=RIGHT)
        )
        self.wait(0.5)

        # Arrow showing imposed swirl bias (e.g. net clockwise swirl)
        swirl_arrow = CurvedArrow(
            start_point=RIGHT * 2.5 + UP * 1.0,
            end_point=RIGHT * 2.5 + DOWN * 1.0,
            angle=-PI / 2,  # draw it roughly clockwise
            color=RED,
            stroke_width=4,
        )
        swirl_label = Text("Net swirl bias", font_size=28).next_to(swirl_arrow, RIGHT)

        self.play(
            GrowArrow(swirl_arrow),
            FadeIn(swirl_label, shift=RIGHT)
        )
        self.wait(0.5)

        # ------------------------------------------------
        # 4. Morph double vortex into single dominant curl
        # ------------------------------------------------
        # New single large vortex in center, oriented to match bias
        single_vortex = Arc(
            radius=1.3,
            start_angle=PI/2,
            angle=-2 * PI,    # clockwise
            color=RED,
            stroke_width=7,
        ).add_tip(tip_length=0.25, tip_width=0.25)

        single_vortex_label = Text(
            "Single curl dominated",
            font_size=30
        ).next_to(circle, UP).shift(DOWN * 0.6)

        # Animation logic:
        # - Strengthen right vortex (rotate / scale / recolor)
        # - Shrink + fade left vortex
        # - Then morph right vortex into single central vortex

        # Step 1: emphasize the "winning" vortex (e.g. right vortex)
        self.play(
            right_vortex.animate.set_color(RED).scale(1.2),
            left_vortex.animate.set_color(BLUE).set_opacity(0.5),
            run_time=1.2
        )

        # Step 2: shrink and fade the weaker vortex
        self.play(
            left_vortex.animate.scale(0.5).set_opacity(0.1),
            run_time=1.0
        )

        # Step 3: transform boosted right vortex into central single vortex
        self.play(
            Transform(right_vortex, single_vortex),
            FadeOut(left_vortex),
            FadeOut(valve),
            FadeOut(swirl_arrow),
            FadeOut(swirl_label),
            FadeOut(dean_label),
            FadeIn(single_vortex_label),
            run_time=1.6
        )

        self.wait(1)

        # ------------------------------------------------
        # 5. Add explanatory text overlay
        # ------------------------------------------------
        explanation_lines = VGroup(
            Text("Dean pair: curvature → symmetric secondary flow", font_size=26),
            Text("Asymmetry adds net swirl / blocks one side", font_size=26),
            Text("One vortex is reinforced, the other is weakened", font_size=26),
            Text("→ Single coherent curl dominates cross-section", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(DOWN + LEFT).shift(UP * 0.4)

        self.play(FadeIn(explanation_lines, lag_ratio=0.2))
        self.wait(2)

        # Final pause
        self.play(
            right_vortex.animate.set_stroke(width=8),
            circle.animate.set_stroke(width=4),
            run_time=1.0
        )
        self.wait(2)
