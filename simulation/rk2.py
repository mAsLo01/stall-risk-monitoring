"""
rk2.py

Численный интегратор второго порядка (метод Рунге–Кутты).

Используется для решения системы дифференциальных уравнений:
- θ (тангаж)
- q (угловая скорость)
- V (скорость)
- h (высота)
- R (риск — обновляется отдельно)
"""

from copy import deepcopy
import numpy as np


class RK2Integrator:
    """
    Реализация RK2 (midpoint method).
    """

    def step(self, state, dynamics, risk_model, u, throttle, dt):
        """
        Выполняет один шаг интегрирования.

        Параметры:
        state — текущее состояние
        dynamics — модель самолёта
        risk_model — модель риска
        u — управление рулём высоты
        T — тяга
        dt — шаг времени

        Возвращает:
        новое состояние
        """

        # =========================
        # ШАГ 1: k1 (начальная производная)
        # =========================

        dtheta1, dq1, dV1, dgamma1, dh1, dx1, dsep1, diag1 = dynamics.derivatives(state, u, throttle, dt)

        # =========================
        # ШАГ 2: промежуточное состояние
        # =========================

        mid_state = deepcopy(state)

        mid_state.theta += 0.5 * dt * dtheta1
        mid_state.q     += 0.5 * dt * dq1
        mid_state.V     += 0.5 * dt * dV1
        mid_state.gamma += 0.5 * dt * dgamma1
        mid_state.h     += 0.5 * dt * dh1
        mid_state.x     += 0.5 * dt * dx1
        mid_state.sep   += 0.5 * dt * dsep1

        # =========================
        # ШАГ 3: k2 (уточнённая производная)
        # =========================

        dtheta2, dq2, dV2, dgamma2, dh2, dx2, dsep2, diag2 = dynamics.derivatives(mid_state, u, throttle, dt)

        # =========================
        # ШАГ 4: обновление состояния
        # =========================

        state.theta += dt * dtheta2
        state.q     += dt * dq2
        state.V     += dt * dV2
        state.gamma += dt * dgamma2
        state.h     += dt * dh2
        state.x += dt * dx2
        state.sep += dt * dsep2
        state.sep = np.clip(state.sep, 0.0, 1.0)
        aero, forces, stall, moment, energy, throttle = diag2
        dynamics.write_diagnostics_to_state(
                state=state,
                aero=aero,
                forces=forces,
                stall=stall,
                moment=moment,
                energy=energy,
                throttle=throttle
            )
        state.gamma = np.clip(state.gamma, -np.pi / 2, np.pi / 2)
        state.theta = np.clip(state.theta, -np.pi / 2, np.pi / 2)

        alpha = dynamics.compute_alpha(state.theta, state.gamma)

        state.alpha_dot = (alpha - state.alpha_prev) / dt
        state.alpha_prev = alpha
        state.alpha = alpha




        # =========================
        # ШАГ 5: обновление риска
        # =========================



        alpha_stall = dynamics.aircraft.alpha_stall_on + dynamics.config.dAlpha_crit

        f = risk_model.compute_f(
            alpha=state.alpha,
            alpha_dot=state.alpha_dot,
            q=state.q,
            speed_margin_1g=state.speed_margin_1g,
            speed_margin_maneuver=state.speed_margin_maneuver,
            sep=state.sep,
            alpha_stall=alpha_stall,
            energy_margin=state.energy_margin,
            vertical_speed=state.vertical_speed
        )

        state.R = risk_model.update(state.R, f, dt)

        state.mode = risk_model.compute_mode(
            R=state.R,
            sep=state.sep,
            speed_margin_1g=state.speed_margin_1g,
            speed_margin_maneuver=state.speed_margin_maneuver,
            alpha=state.alpha,
            alpha_stall=alpha_stall
        )

        return state