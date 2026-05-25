import numpy as np
from config import risk_decay_time


class RiskModel:
    """
    Интегральная модель риска сваливания.

    Риск растёт при:
    - приближении угла атаки к критическому;
    - быстром росте угла атаки;
    - малом запасе скорости V / Vstall;
    - развитии отрыва потока sep;
    - большой угловой скорости тангажа q.

    R — накопленный риск.
    """

    def __init__(self, rise_time=0.6, decay_time=risk_decay_time):
        self.rise_time = rise_time
        self.decay_time = decay_time

    def compute_f(
            self,
            alpha,
            alpha_dot,
            q,
            speed_margin_1g,
            speed_margin_maneuver,
            sep,
            alpha_stall,
            energy_margin=0.0,
            vertical_speed=0.0
    ):
        """
        Мгновенная функция опасности.
        """

        # 1. Риск по углу атаки
        # Начинаем учитывать не только после stall,
        # а уже при достижении примерно 70% критического угла.
        alpha_ratio = abs(alpha) / max(alpha_stall, 1e-6)
        alpha_term = np.clip((alpha_ratio - 0.7) / 0.3, 0.0, 1.5)

        # 2. Риск по скорости роста угла атаки
        # Если alpha быстро растёт, риск должен увеличиваться заранее.
        # CESSNA_172
        # alpha_dot_term = np.clip(max(0.0, alpha_dot) / 1.0, 0.0, 1.0)
        # 737-800
        alpha_dot_term = np.clip(max(0.0, alpha_dot) / 0.4, 0.0, 1.0)

        # 3. Риск по запасу скорости
        # speed_margin = V / Vstall.
        # Выше 1.3 — нормально, около 1.0 — опасно.

        speed_margin_used = min(speed_margin_1g, speed_margin_maneuver)

        speed_term = np.clip((1.5 - speed_margin_used) / 0.5, 0.0, 2.0)

        # 4. Риск по степени отрыва потока
        sep_term = np.clip(sep, 0.0, 1.0)

        # 5. Риск по угловой скорости тангажа
        # CESSNA_172
        # q_term = np.clip(abs(q) / 0.8, 0.0, 1.0)

        #737-800
        q_term = np.clip(abs(q) / 0.05, 0.0, 1.0)

        energy_term = np.clip((-energy_margin - 0.2) / 0.8, 0.0, 1.5)

        climb_factor = np.clip(vertical_speed / 5.0, 0.0, 1.0)
        climb_energy_term = energy_term * climb_factor

        f = (
                0.28 * alpha_term
                + 0.30 * speed_term
                + 0.22 * sep_term
                + 0.08 * alpha_dot_term
                + 0.04 * q_term
                + 0.18 * climb_energy_term
        )
        if sep > 0.5:
            f += 0.5 * (sep - 0.5) / 0.5

        if speed_margin_used < 1.15:
            f += 0.7 * (1.15 - speed_margin_used) / 0.15

        if alpha_ratio > 0.9:
            f += 0.5 * (alpha_ratio - 0.9) / 0.2
        return f

    def update(self, R, f, dt):
        R_target = np.clip(3.0 * f, 0.0, 10.0)

        if R_target > R:
            tau = self.rise_time
        else:
            tau = self.decay_time

        R_new = R + dt * (R_target - R) / tau
        return np.clip(R_new, 0.0, 10.0)

    def compute_mode(
            self,
            R,
            sep,
            speed_margin_1g,
            speed_margin_maneuver,
            alpha,
            alpha_stall
    ):
        alpha_ratio = abs(alpha) / max(alpha_stall, 1e-6)

        speed_margin_used = min(speed_margin_1g, speed_margin_maneuver)

        if sep > 0.7 or alpha_ratio > 1.15:
            return "STALL"

        if speed_margin_used < 1.0:
            return "LOW_SPEED"

        if R > 2.0 or speed_margin_used < 1.1 or alpha_ratio > 1.0:
            return "WARNING"

        if R > 1.0 or speed_margin_used < 1.3 or alpha_ratio > 0.8:
            return "CAUTION"

        return "NORMAL"