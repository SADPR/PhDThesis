from manimlib import *

class Scene1(Scene):
    def construct(self):
        # Colors
        red = RED_E
        blue = BLUE_E
        encoder_color = PURPLE_B
        decoder_color = PURPLE_B

        # Input and output layer dots (red)
        input_dots = VGroup(*[Dot(color=red).shift(UP * i * 0.5) for i in range(3, -2, -1)])
        output_dots = VGroup(*[Dot(color=red).shift(UP * i * 0.5) for i in range(3, -2, -1)])
        latent_dots = VGroup(*[Dot(color=blue).shift(UP * i * 0.5) for i in range(1, -2, -1)])

        # Positioning
        input_dots.move_to(LEFT * 4)
        output_dots.move_to(RIGHT * 4)
        latent_dots.move_to(ORIGIN)

        # Ovals around layers
        input_oval = Ellipse(width=0.8, height=2.0, color=WHITE).move_to(input_dots)
        output_oval = Ellipse(width=0.8, height=2.0, color=WHITE).move_to(output_dots)
        latent_oval = Ellipse(width=0.6, height=2.0, color=WHITE).move_to(latent_dots)

        # Encoder and decoder blocks
        encoder = Polygon(
            input_dots.get_right() + UP * 1.0,
            latent_dots.get_left() + UP * 0.5,
            latent_dots.get_left() + DOWN * 0.5,
            input_dots.get_right() + DOWN * 1.0,
            color=encoder_color, fill_opacity=0.3
        )

        decoder = Polygon(
            latent_dots.get_right() + UP * 0.5,
            output_dots.get_left() + UP * 1.0,
            output_dots.get_left() + DOWN * 1.0,
            latent_dots.get_right() + DOWN * 0.5,
            color=decoder_color, fill_opacity=0.3
        )

        # Labels
        u_label = Tex(r"\mathbf{u}").next_to(input_dots, UP)
        q_label = Tex(r"\mathbf{q}").next_to(latent_dots, UP)
        uhat_label = Tex(r"\hat{\mathbf{u}}").next_to(output_dots, UP)

        phiT_label = Tex(r"\boldsymbol{\Phi}^\top").move_to(encoder.get_center())
        phi_label = Tex(r"\boldsymbol{\Phi}").move_to(decoder.get_center())

        # Add everything to the scene
        self.add(input_dots, output_dots, latent_dots)
        self.add(input_oval, output_oval, latent_oval)
        self.add(encoder, decoder)
        self.add(u_label, q_label, uhat_label)
        self.add(phiT_label, phi_label)

