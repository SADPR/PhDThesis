from manim import *
import numpy as np

class MeshElementAssembly(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # ---------- Titles & static top panels ----------
        title_line = Text("PROM Assembly").scale(0.6)
        title = VGroup(title_line).arrange(DOWN, center=True).to_edge(UP)
        self.play(FadeIn(title, shift=UP*0.4), run_time=0.5)
        title.set_color(BLACK)

        # ---------- Mesh (top) ----------
        nx, ny = 4, 3
        W, H   = 5.0, 2.5
        mesh_color, hi_color = GRAY, DARK_BLUE

        domain = Rectangle(width=W, height=H, stroke_color=BLACK)
        x0, y0 = domain.get_left()[0], domain.get_bottom()[1]
        dx, dy = W/nx, H/ny

        def node_xy(i, j):
            return np.array([x0 + i*dx, y0 + j*dy, 0])

        vlines = [Line(node_xy(i,0), node_xy(i,ny), stroke_color=mesh_color) for i in range(nx+1)]
        hlines = [Line(node_xy(0,j), node_xy(nx,j), stroke_color=mesh_color) for j in range(ny+1)]

        tris = []
        for i in range(nx):
            for j in range(ny):
                p00,p10,p01,p11 = node_xy(i,j), node_xy(i+1,j), node_xy(i,j+1), node_xy(i+1,j+1)
                tris += [
                    Polygon(p00,p10,p01, color=mesh_color, stroke_width=2),
                    Polygon(p11,p01,p10, color=mesh_color, stroke_width=2)
                ]

        tri_group  = VGroup(*tris)
        mesh_group = VGroup(domain, *vlines, *hlines, tri_group).scale(0.75).to_edge(UP, buff=0.5).shift(DOWN*0.5 + RIGHT*0.0)
        hi = tris[0].copy().set_fill(hi_color, opacity=0.4).set_stroke(hi_color, width=3)

        # ---------- Colors ----------
        phi_c, r_c, g_c = RED, GREEN, RED  # final formulas will be black; ECM diffs in red

        # ---------- Helpers ----------
        def Phi_matrix_e():
            rows = [[r"\phi_{11}^{e}", r"\phi_{12}^{e}", r"\phi_{13}^{e}"],
                    [r"\phi_{21}^{e}", r"\phi_{22}^{e}", r"\phi_{23}^{e}"]]
            return Matrix(rows, h_buff=1.0, v_buff=0.95).scale(0.9).set_color(phi_c)

        def Phi_matrix_k(k):
            rows = [[f"\\phi_{{11}}^{{{k}}}", f"\\phi_{{12}}^{{{k}}}", f"\\phi_{{13}}^{{{k}}}"],
                    [f"\\phi_{{21}}^{{{k}}}", f"\\phi_{{22}}^{{{k}}}", f"\\phi_{{23}}^{{{k}}}"]]
            return Matrix(rows, h_buff=1.0, v_buff=0.95).scale(0.9).set_color(phi_c)

        def R_vector_e():
            rows = [[r"R_{1}^{e}"], [r"R_{2}^{e}"], [r"R_{3}^{e}"]]
            return Matrix(rows, v_buff=0.95).scale(0.9).set_color(r_c)

        def R_vector_k(k):
            rows = [[f"R_{{1}}^{{{k}}}"], [f"R_{{2}}^{{{k}}}"], [f"R_{{3}}^{{{k}}}"]]
            return Matrix(rows, v_buff=0.95).scale(0.9).set_color(r_c)

        # Global RHS vector (no label here; we'll use a label only in the formula row)
        def RHS_vector_only():
            return Matrix([[r"(\Phi^{T}R)_1"], [r"(\Phi^{T}R)_2"]],
                          v_buff=0.95).scale(0.9).set_color(g_c)

        # ---------- Show mesh ----------
        self.play(Create(mesh_group))
        self.play(Write(hi), run_time=0.5)
        self.wait(5.0)

        # Left: PROM (all elements) — initially emphasized (opacity 1.0)
        left_line1 = Text("PROM assembly").scale(0.4)
        left_line2 = Text("(all elements)").scale(0.4)
        left_subtitle = VGroup(left_line1, left_line2).arrange(DOWN, center=True)
        left_formula  = MathTex(r"\Phi^{T}R \;=\; \sum_{e=1}^{N_e} \Phi_{e}^{T} R_{e}").scale(0.9)
        # Build the block normally
        left_block = VGroup(left_subtitle, left_formula).arrange(DOWN, buff=0.15)

        # Save the final position (with to_edge + shift)
        final_pos = left_block.copy().to_edge(UL).shift(DOWN*0.3 + RIGHT*0.0)

        # Reset the actual block to the origin (start here)
        left_block.move_to(ORIGIN).shift(DOWN*0.5)
        left_subtitle.set_color(BLACK)
        left_formula.set_color_by_tex(r"\Phi^{T}R", BLACK)
        left_formula.set_color_by_tex(r"=", BLACK)
        left_formula.set_color_by_tex(r"\sum_{e=1}^{N_e}", BLACK)
        left_formula.set_color_by_tex(r"\Phi_{e}^{T}", BLACK)
        left_formula.set_color_by_tex(r"R_{e}", BLACK)

        self.play(FadeIn(left_block), run_time=0.5)
        self.wait(8.0)
        self.play(left_block.animate.move_to(final_pos), run_time=0.7)

        # ---------- Build initial (general e) ----------
        # Top FORMULA row:  Φ^T R  +=  Φ_e^T  R_e
        RHS_lbl = MathTex(r"\Phi^{T}R",  color=g_c)
        Phi_lbl = MathTex(r"\Phi_e^{T}", color=phi_c)
        Re_lbl  = MathTex(r"R_e",        color=r_c)

        # Bottom MATRICES row:  [RHS vec]  +=  [Φ_e^T]  [R_e]
        RHS_vec  = RHS_vector_only()
        plus_mat = MathTex("+=", color=g_c).scale(0.9)
        PhiM     = Phi_matrix_e()
        RV       = R_vector_e()

        RHS_lbl.move_to(ORIGIN)
        Phi_lbl.move_to(ORIGIN).move_to(LEFT * 2.5)   # slight nudge
        Re_lbl.move_to(ORIGIN).move_to(RIGHT * 2.5)
        RHS_vec.move_to(ORIGIN)
        plus_mat.move_to(ORIGIN)
        PhiM.move_to(ORIGIN).shift(LEFT*2.5 + DOWN*2)
        RV.move_to(ORIGIN).shift(RIGHT*2.5 + DOWN*2)

        # Hide RHS (both rows) and '+=' initially; reveal at k=1
        RHS_lbl.set_opacity(0)
        RHS_vec.set_opacity(0)
        plus_mat.set_opacity(0)

        # show only Φ_e^T, R_e (both rows start with e)
        self.play(Write(Phi_lbl), Write(Re_lbl), Write(PhiM), Write(RV))
        self.wait(10.0)

        # --- animate the formula-row items to your final positions ---
        for m, pos in [
            (RHS_lbl,   LEFT * 4.4),
            (Phi_lbl,   RIGHT * 1.5),
            (Re_lbl,    RIGHT * 4.85),
            (RHS_vec,   LEFT * 4.4 + DOWN * 2),
            (plus_mat,  LEFT * 1.9 + DOWN * 2),
            (PhiM,      RIGHT * 1.5 + DOWN * 2),
            (RV,        RIGHT * 4.85 + DOWN * 2)
        ]:
            m.generate_target()
            m.target.move_to(pos)

        self.play(
            *[MoveToTarget(m) for m in (RHS_lbl, Phi_lbl, Re_lbl, RHS_vec, plus_mat, PhiM, RV)],
            run_time=0.7, rate_func=smooth
        )

        # save the final positions for RHS labels
        RHS_lbl.save_state()
        RHS_vec.save_state()

        # ---------- Loop through elements ----------
        for k, tri in enumerate(tris[:], start=1):
            self.play(
                hi.animate.become(tri.copy().set_fill(hi_color, opacity=0.4).set_stroke(hi_color, width=3)),
                run_time=0.35
            )
            # update element index in both rows
            self.play(
                Transform(Phi_lbl, MathTex(f"\\Phi_{{{k}}}^T", color=phi_c).move_to(Phi_lbl)),
                Transform(Re_lbl,  MathTex(f"R_{{{k}}}",       color=r_c  ).move_to(Re_lbl)),
                Transform(PhiM,    Phi_matrix_k(k).move_to(PhiM)),
                Transform(RV,      R_vector_k(k).move_to(RV)),
                run_time=0.2
            )

            if k == 1:
                # reveal global RHS and '+=' smoothly from the left
                self.add(RHS_lbl, RHS_vec, plus_mat)
                self.play(
                    Restore(RHS_lbl), Restore(RHS_vec),
                    RHS_lbl.animate.set_opacity(1),
                    RHS_vec.animate.set_opacity(1),
                    plus_mat.animate.set_opacity(1),
                    run_time=0.7, rate_func=smooth
                )
            else:
                # subtle cue on subsequent elements
                self.play(
                    Indicate(RHS_lbl, color=g_c, scale_factor=1.03),
                    Indicate(RHS_vec, color=g_c, scale_factor=1.03),
                    run_time=0.1
                )

        self.wait(0.8)
        self.play(FadeOut(hi), run_time=0.4)

        # Hide the matrix row (RHS_vec, plus_mat, PhiM, RV) smoothly
        bottom_matrix_group = VGroup(RHS_vec, plus_mat, PhiM, RV)
        self.play(FadeOut(bottom_matrix_group, shift=DOWN*0.3), run_time=0.5)

        sum_all = MathTex(
            r"\Phi^{T}R \;=\; \sum_{e=1}^{N_e} \Phi_{e}^{T} R_{e}",
            substrings_to_isolate=[r"\Phi^{T}R", r"=", r"\sum_{e=1}^{N_e}", r"\Phi_{e}^{T}", r"R_{e}", r"N_e"]
        ).scale(1.0)

        # keep final in black (your preference)
        sum_all.set_color_by_tex(r"\Phi^{T}R", BLACK)
        sum_all.set_color_by_tex(r"\Phi_{e}^{T}", BLACK)
        sum_all.set_color_by_tex(r"R_{e}", BLACK)
        sum_all.set_color_by_tex(r"=", BLACK)
        sum_all.set_color_by_tex(r"\sum_{e=1}^{N_e}", BLACK)

        sum_all.move_to(VGroup(RHS_lbl, Phi_lbl, Re_lbl).get_center())

        top_row_group = VGroup(RHS_lbl, Phi_lbl, Re_lbl)
        self.play(ReplacementTransform(top_row_group, sum_all), run_time=0.7, rate_func=smooth)
        self.wait(5.0)

        sum_all_big = MathTex(
            r"\Phi^{T}R \;=\; \sum_{e=1}^{\text{50,000,000}} \Phi_{e}^{T} R_{e}",
            substrings_to_isolate=[r"\Phi^{T}R", r"=", r"\sum_{e=1}^{\text{50,000,000}}", r"\Phi_{e}^{T}", r"R_{e}", r"\text{50,000,000}"]
        ).scale(1.0).move_to(sum_all)

        sum_all_big.set_color_by_tex(r"\Phi^{T}R", BLACK)
        sum_all_big.set_color_by_tex(r"\Phi_{e}^{T}", BLACK)
        sum_all_big.set_color_by_tex(r"R_{e}", BLACK)
        sum_all_big.set_color_by_tex(r"=", BLACK)
        sum_all_big.set_color_by_tex(r"\sum_{e=1}^{\text{50,000,000}}", RED)

        # Everything stays black (no highlights)
        self.play(TransformMatchingTex(sum_all, sum_all_big), run_time=1.0)
        self.wait(8.0)

        # continue using the new object
        sum_all = sum_all_big

        # 1) Prepare the ECM formula *at the same spot* as sum_all (no movement yet)
        right_formula = MathTex(
            r"\Phi^{T}R \;\approx\; \sum_{e\in\tilde{\mathcal{E}}} \xi_e\, \Phi_{e}^{T} R_{e}",
            substrings_to_isolate=[r"\Phi^{T}R", r"\approx", r"\sum_{e\in\tilde{\mathcal{E}}}", r"\xi_e", r"\Phi_{e}^{T}", r"R_{e}"]
        ).scale(1.0).move_to(sum_all)

        # Highlight only the differences in red
        right_formula.set_color_by_tex(r"\Phi^{T}R", BLACK)
        right_formula.set_color_by_tex(r"\approx", RED)
        right_formula.set_color_by_tex(r"\sum_{e\in\tilde{\mathcal{E}}}", RED)
        right_formula.set_color_by_tex(r"\xi_e", RED)
        right_formula.set_color_by_tex(r"\Phi_{e}^{T}", BLACK)
        right_formula.set_color_by_tex(r"R_{e}", BLACK)

        # 2) Do the morph ALONE (no other animations in the same self.play)
        self.play(TransformMatchingTex(sum_all, right_formula), run_time=0.8)
        self.wait(5.0)

        # 3) Rebind: sum_all is now the right formula (so we keep working with the same object)
        sum_all = right_formula

        # 4) Create the right subtitle and assemble the block *after* the morph
        right_line1 = Text("HPROM assembly").scale(0.4)
        right_line2 = Text("(ECM subset + weights)").scale(0.4)
        right_subtitle = VGroup(right_line1, right_line2).arrange(DOWN, center=True)
        right_subtitle.set_color(BLACK)

        # Place the subtitle just below the current formula, then group
        right_subtitle.next_to(sum_all, UP, buff=0.15)
        right_block = VGroup(right_subtitle, sum_all)

        self.play(right_block.animate.move_to(DOWN * 0.5), run_time=0.5)
        self.wait(5.0)

        # 5) Now move the whole block to its final UR location and ghost the left panel
        final_pos_right_block = right_block.copy().to_edge(UR).shift(DOWN*0.3 + RIGHT*0.2)

        self.play(
            left_block.animate.set_opacity(0.2),
            right_block.animate.move_to(final_pos_right_block),
            run_time=0.7
        )

        # ---------- ECM subset selection & weighted mini-assembly (matrix-style) ----------
        # 1) Dim all elements
        self.play(*[
            t.animate.set_fill(mesh_color, opacity=0.12).set_stroke(mesh_color, width=2)
            for t in tris
        ], run_time=0.4)

        # 2) Choose subset and highlight (ECM subset \tilde{E})
        subset_idx = [0, 5, 8, 17]   # your picked elements
        subset_overlays = [
            tris[i].copy().set_fill(DARK_GRAY, opacity=0.80).set_stroke(DARK_GRAY, width=3)
            for i in subset_idx
        ]
        subset_vg = VGroup(*subset_overlays)
        self.play(FadeIn(subset_vg, scale=1.02), run_time=0.6)
        self.wait(8.0)

        # 3) Two-row mini assembly:  Φ^T R  +=  ξ_e · Φ_e^T · R_e
        #    Top FORMULA row (labels)
        RHS_lbl_E = MathTex(r"\Phi^{T}R", color = RED)
        Phi_lbl_E = MathTex(r"\Phi_{e}^{T}", color = RED)
        Re_lbl_E  = MathTex(r"R_{e}", color = GREEN)

        #    Bottom MATRICES row (objects)
        RHS_vec_E = RHS_vector_only()
        plus_matE = MathTex("+=", color=RED).scale(0.9)
        Xi_scalar = MathTex(r"\xi_e", color=RED).scale(0.9)  # will transform each iteration
        PhiM_E    = Phi_matrix_e()
        RV_E      = R_vector_e()

        for m in (RHS_lbl_E, RHS_vec_E, plus_matE):
            m.save_state()

        RHS_lbl_E.move_to(ORIGIN)
        Phi_lbl_E.move_to(ORIGIN).move_to(LEFT * 2.5)   # slight nudge
        Re_lbl_E.move_to(ORIGIN).move_to(RIGHT * 2.5)
        RHS_vec_E.move_to(ORIGIN)
        plus_matE.move_to(ORIGIN)
        Xi_scalar.move_to(ORIGIN)
        PhiM_E.move_to(ORIGIN).shift(LEFT*2.5 + DOWN*2)
        RV_E.move_to(ORIGIN).shift(RIGHT*2.5 + DOWN*2)

        # Hide RHS (both rows) and '+=' initially; reveal at first weighted term
        for m in (RHS_lbl_E, RHS_vec_E, plus_matE):
            m.set_opacity(0)

        self.play(Write(Phi_lbl_E), Write(Re_lbl_E), Write(Xi_scalar), Write(PhiM_E), Write(RV_E))
        self.wait(6.0)

        # --- animate the formula-row items to your final positions ---
        for m, pos in [
            (RHS_lbl_E,   LEFT * 4.4),
            (Phi_lbl_E,   RIGHT * 1.5),
            (Re_lbl_E,    RIGHT * 4.85),
            (RHS_vec_E,   LEFT * 4.4 + DOWN * 2),
            (Xi_scalar, LEFT * 0.5 + DOWN * 2),
            (plus_matE,  LEFT * 1.9 + DOWN * 2),
            (PhiM_E,      RIGHT * 1.5 + DOWN * 2),
            (RV_E,        RIGHT * 4.85 + DOWN * 2)
        ]:
            m.generate_target()
            m.target.move_to(pos)

        self.play(
            *[MoveToTarget(m) for m in (RHS_lbl_E, Phi_lbl_E, Re_lbl_E, RHS_vec_E, Xi_scalar, plus_matE, PhiM_E, RV_E)],
            run_time=0.7, rate_func=smooth
        )

        # Prepare a persistent group of the per-term factors so we can transform them each loop
        Xi_scalar.save_state()
        Phi_lbl_E.save_state(); Re_lbl_E.save_state()
        PhiM_E.save_state(); RV_E.save_state()

        # 4) Iterate only over the subset: highlight element, update (ξ, Φ_e^T, R_e), then cue accumulation
        for vis_id, tri_idx in enumerate(subset_idx, start=1):
            # Focus outline on the currently selected element
            focus = tris[tri_idx].copy().set_stroke(RED, width=4).set_fill(DARK_GRAY, opacity=0.0)
            self.play(FadeIn(focus), run_time=0.35)

            # Build current labels/matrices
            xi_now_mat = MathTex(fr"\xi_{{{tri_idx+1}}}", color=RED).scale(0.9).move_to(Xi_scalar)

            Phi_now_lbl = MathTex(fr"\Phi_{{{tri_idx+1}}}^T", color=RED).move_to(Phi_lbl_E)
            Re_now_lbl  = MathTex(fr"R_{{{tri_idx+1}}}", color=GREEN).move_to(Re_lbl_E)

            Phi_now_mat = Phi_matrix_k(tri_idx+1).move_to(PhiM_E)
            Re_now_mat  = R_vector_k(tri_idx+1).move_to(RV_E)

            # First element: reveal RHS and '+=' smoothly
            if vis_id == 1:
                # self.add(RHS_lbl_E, RHS_vec_E, plus_matE)
                self.play(
                    Restore(RHS_lbl_E), Restore(RHS_vec_E),
                    RHS_lbl_E.animate.set_opacity(1),
                    RHS_vec_E.animate.set_opacity(1),
                    plus_matE.animate.set_opacity(1),
                    run_time=0.5
                )

            # Write/transform the weighted factors into place (both rows)
            self.play(
                Transform(Phi_lbl_E, Phi_now_lbl),
                Transform(Re_lbl_E,  Re_now_lbl),
                Transform(Xi_scalar, xi_now_mat),
                Transform(PhiM_E,    Phi_now_mat),
                Transform(RV_E,      Re_now_mat),
                run_time=0.35
            )

            # Subtle cue on the HPROM formula (right panel)
            self.play(Indicate(sum_all, color=RED, scale_factor=1.02), run_time=0.18)

            # Optional: brief “tap” on RHS to suggest accumulation
            self.play(
                Indicate(RHS_lbl_E, color=RED, scale_factor=1.03),
                Indicate(RHS_vec_E, color=RED, scale_factor=1.03),
                run_time=0.12
            )

            self.play(FadeOut(focus), run_time=0.18)
        
        self.wait(2.0)

        # Fade out the mini-assembly rows (optional)
        self.play(
            *[FadeOut(m) for m in (RHS_lbl_E, Phi_lbl_E, Re_lbl_E,
                                RHS_vec_E, plus_matE, Xi_scalar, PhiM_E, RV_E)],
            FadeOut(subset_vg),
            run_time=0.5
        )

        # Final big HPROM summary (center screen)
        final_sum = MathTex(
            r"\Phi^{T}R \;=\; \sum_{e=1}^{N_e} \Phi_{e}^{T} R_{e}\;\approx\; \sum_{e\in\tilde{\mathcal{E}}} \xi_e\, \Phi_{e}^{T} R_{e}",
        ).scale(1.05).set_color(BLACK).move_to(ORIGIN)

        self.play(Write(final_sum), run_time=0.8)
        self.wait(6.0)




