import matplotlib
matplotlib.use("TkAgg")

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np




class Visualizer:
    def __init__(self, aircraft, config):
        plt.ion()

        self.aircraft = aircraft
        self.config = config

        self.alpha_stall_on = aircraft.alpha_stall_on + config.dAlpha_crit
        self.alpha_stall_off = aircraft.alpha_stall_off + config.dAlpha_crit

        self.fig = plt.figure(figsize=(14, 8))

        gs = self.fig.add_gridspec(
            5, 2,
            width_ratios=[2.2, 1.2],
            height_ratios=[1, 1, 1, 1, 1]
        )

        self.ax = self.fig.add_subplot(gs[:, 0])

        self.ax_alpha = self.fig.add_subplot(gs[0, 1])
        self.ax_CL = self.fig.add_subplot(gs[1, 1])
        self.ax_margin = self.fig.add_subplot(gs[2, 1])
        self.ax_sep = self.fig.add_subplot(gs[3, 1])
        self.ax_R = self.fig.add_subplot(gs[4, 1])

        self.t_hist = []
        self.alpha_hist = []
        self.CL_ratio_hist = []
        self.sep_hist = []
        self.R_hist = []

        self.alpha_line, = self.ax_alpha.plot([], [])
        self.CL_line, = self.ax_CL.plot([], [])
        self.margin_1g_hist = []
        self.margin_maneuver_hist = []

        self.margin_1g_line, = self.ax_margin.plot([], [], label="V/Vs 1g")
        self.margin_m_line, = self.ax_margin.plot([], [], label="V/Vs maneuver")

        self.sep_line, = self.ax_sep.plot([], [])
        self.R_line, = self.ax_R.plot([], [])

        self.ax_margin.legend(
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            frameon=True
        )



        self.ax_alpha.axhline(
            np.degrees(self.alpha_stall_on),
            linestyle="--",
            linewidth=1,
            label="stall on"
        )

        self.ax_alpha.axhline(
            np.degrees(self.alpha_stall_off),
            linestyle=":",
            linewidth=1,
            label="stall off"
        )

        self.ax_margin.axhline(1.3, linestyle=":", linewidth=1)
        self.ax_margin.axhline(1.0, linestyle="--", linewidth=1)

        self.ax_sep.axhline(0.7, linestyle="--", linewidth=1)
        self.ax_R.axhline(1.0, linestyle=":", linewidth=1)
        self.ax_R.axhline(2.0, linestyle="--", linewidth=1)

        self.ax_CL.axhline(1.0, linestyle="--", linewidth=1)

        self.ax_alpha.set_ylabel("α, deg")
        self.ax_CL.set_ylabel("CL/Clmax")
        self.ax_margin.set_ylabel("V/Vs")
        self.ax_sep.set_ylabel("sep")
        self.ax_R.set_ylabel("R")
        self.ax_R.set_xlabel("t, s")

        for ax in [self.ax_alpha, self.ax_CL, self.ax_margin, self.ax_sep, self.ax_R]:
            ax.grid(True)

        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(-20, 120)
        self.ax.set_aspect("equal")

        # =====================
        # ГОРИЗОНТ
        # =====================
        self.horizon_line, = self.ax.plot(
            [-200, 200], [0, 0],
            color="black",
            lw=2
        )


        # =====================
        # САМОЛЁТ
        # =====================
        self.plane = plt.Polygon(self._plane_shape(), closed=True)
        self.ax.add_patch(self.plane)

        # =====================
        # ТРАЕКТОРИЯ
        # =====================
        self.trail_x = []
        self.trail_y = []
        self.trail_line, = self.ax.plot([], [], color="blue", lw=1)

        # =====================
        # ВЕКТОР СКОРОСТИ (γ)
        # =====================
        self.vel_line, = self.ax.plot([], [], "--", color="orange")

        # =====================
        # ВЕКТОР ТАНГАЖА (θ)
        # =====================
        self.theta_line, = self.ax.plot([], [], "--", color="green")

        self.info_text = self.ax.text(
            0.02, 0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
        )


    # ---------------------

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    def _plane_shape(self):
        return np.array([
            [6, 0],
            [2, 1],
            [-2, 0.5],
            [-5, 2],
            [-3, 0],
            [-5, -2],
            [-2, -0.5],
            [2, -1],
        ])

    def rotate(self, pts, angle):
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        return pts @ R.T

    # ---------------------
    def get_color(self, R, mode="NORMAL"):
        if mode == "STALL":
            return "#6a0dad"  # purple
        if R < 0.5:
            return "#2ca02c"
        elif R < 2:
            return "#ff7f0e"
        else:
            return "#d62728"

    # ---------------------
    def update(self, state, t):
        theta = state.theta
        gamma = state.gamma
        h = state.h
        V = state.V
        R = state.R

        scale = 0.2

        x = state.x
        y = h

        # =====================
        # САМОЛЁТ
        # =====================
        shape = self.rotate(self._plane_shape(), theta)
        shape[:, 0] += x
        shape[:, 1] += y

        self.plane.set_xy(shape)
        self.plane.set_color(self.get_color(R, state.mode))

        # =====================
        # ТРАЕКТОРИЯ
        # =====================
        self.trail_x.append(state.x)
        self.trail_y.append(y)
        self.trail_line.set_data(self.trail_x, self.trail_y)

        # =====================
        # ВЕКТОР СКОРОСТИ (γ)
        # =====================
        vx = V * np.cos(gamma) * scale
        vy = V * np.sin(gamma) * scale

        self.vel_line.set_data(
            [x, x + vx],
            [y, y + vy]
        )

        # =====================
        # ВЕКТОР ТАНГАЖА (θ)
        # =====================
        tx = np.cos(theta) * 10
        ty = np.sin(theta) * 10

        self.theta_line.set_data(
            [x, x + tx],
            [y, y + ty]
        )
        CL_ratio = getattr(state, "CL_ratio", 0.0)

        alpha = theta - gamma
        alpha_deg = np.degrees(alpha)
        gamma_deg = np.degrees(gamma)
        theta_deg = np.degrees(theta)

        self.ax.set_title(f"Stall simulation | mode={state.mode}")
        self.info_text.set_text(
            f"α = {alpha_deg:.1f}°\n"
            f"αdot = {np.degrees(state.alpha_dot):.1f}°/s\n"
            f"θ = {theta_deg:.1f}°\n"
            f"γ = {gamma_deg:.1f}°\n"
            f"V = {state.V:.1f} m/s\n"
            f"Vs = {state.V_stall:.1f} m/s\n"
            f"V/Vs = {state.speed_margin:.2f}\n"
            f"CL = {state.CL:.2f}\n"
            f"CD = {state.CD:.2f}\n"
            f"sep = {state.sep:.2f}\n"
            f"stall warning = {state.stall_warning}\n"
            f"warning margin = {state.stall_warning_margin:.1f} m/s\n"
            f"R = {state.R:.2f}"
        )

        # =====================
        # ДУГА УГЛА АТАКИ
        # =====================
        radius = 8
        center = (x, y)

        if alpha_deg >= 0:
            # Положительный угол – против часовой стрелки
            theta1 = np.radians(gamma_deg)
            theta2 = np.radians(gamma_deg + alpha_deg)
            angles = np.linspace(theta1, theta2, num=50)
        else:
            # Отрицательный угол – по часовой стрелке
            theta1 = np.radians(gamma_deg)
            theta2 = np.radians(gamma_deg + alpha_deg)  # alpha_deg < 0, поэтому theta2 < theta1
            angles = np.linspace(theta1, theta2, num=50)

        arc_x = center[0] + radius * np.cos(angles)
        arc_y = center[1] + radius * np.sin(angles)
        verts = np.column_stack([arc_x, arc_y])
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(arc_x) - 1)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, color='purple', lw=2, fill=False)

        # Удаляем старые PathPatch
        for old_patch in self.ax.patches:
            if isinstance(old_patch, mpatches.PathPatch):
                old_patch.remove()
        self.ax.add_patch(patch)
        # =====================
        # КАМЕРА
        # =====================
        self.ax.set_xlim(x - 50, x + 50)
        self.ax.set_ylim(y - 50, y + 50)


        # =====================
        # ГОРИЗОНТ
        # =====================
        self.horizon_line.set_data(
            [x - 200, x + 200],
            [0, 0]
        )

        self.t_hist.append(t)
        self.alpha_hist.append(np.degrees(alpha))
        self.CL_ratio_hist.append(CL_ratio)
        self.sep_hist.append(state.sep)
        self.R_hist.append(state.R)

        window = 20.0
        t_min = max(0.0, t - window)
        t_max = max(window, t)

        self.alpha_line.set_data(self.t_hist, self.alpha_hist)
        self.CL_line.set_data(self.t_hist, self.CL_ratio_hist)
        self.margin_1g_hist.append(state.speed_margin_1g)
        self.margin_maneuver_hist.append(state.speed_margin_maneuver)

        self.margin_1g_line.set_data(self.t_hist, self.margin_1g_hist)
        self.margin_m_line.set_data(self.t_hist, self.margin_maneuver_hist)

        self.sep_line.set_data(self.t_hist, self.sep_hist)
        self.R_line.set_data(self.t_hist, self.R_hist)

        for ax in [self.ax_alpha, self.ax_CL, self.ax_margin, self.ax_sep, self.ax_R]:
            ax.set_xlim(t_min, t_max)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # =====================
        # РЕНДЕР
        # =====================
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()