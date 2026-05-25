"""
main.py

Главный файл симуляции динамики сваливания самолёта.

Объединяет:
- AircraftModel (тип самолёта)
- AircraftDynamics (физика движения)
- RiskModel (оценка риска)
- RK2Integrator (численное решение)

Режим: базовый (без UI, фиксированное управление)
"""

import numpy as np

from config import dt
from model import dynamics as dyn
from model.dynamics import AircraftDynamics
from model.risk import RiskModel
from simulation.rk2 import RK2Integrator
from visualization import Visualizer
#from experiments.scenarios import SCENARIOS
from experiments.scenarios_737 import SCENARIOS_737
from AircraftModel import BOEING_737_JSBSIM, CLEAN, LANDING




SCENARIO_NAME = "level_flight_trimmed_737"
#SCENARIO_NAME = "low_speed_clean"




# ГЛАВНАЯ ФУНКЦИЯ


def run_simulation():

    # --- инициализация ---
    scenario = next(s for s in SCENARIOS_737 if s.name == SCENARIO_NAME)

    aircraft = BOEING_737_JSBSIM
    config = scenario.config
    dyn.ELEVATOR_EFFECT_MULTIPLIER = 1.0
    dynamics = AircraftDynamics(aircraft, config)
    risk_model = RiskModel()
    integrator = RK2Integrator()
    visualizer = Visualizer(aircraft, config)

    state = scenario.make_initial_state()





    t = 0.0

    print("=== Simulation started ===")


    # ОСНОВНОЙ ЦИКЛ


    for step in range(2000):

        # управление
        u, throttle = scenario.control_law(t, state)

        # шаг интегрирования
        state = integrator.step(
            state,
            dynamics,
            risk_model,
            u,
            throttle,
            dt
        )

        # текущий угол атаки
        alpha = dynamics.compute_alpha(state.theta, state.gamma)

        # вывод (можно заменить на лог/график)
        print(
            f"t={t:.2f} | "
            f"θ={state.theta:.3f} | "
            f"α={alpha:.3f} | "
            f"gamma={state.gamma:.3f} | "
            f"V={state.V:.2f} | "
            f"h={state.h:.1f} | "
            f"dCLt={state.dCL_turb:.2f} | "
            f"dCmt={state.dCm_turb:.2f} | "
            f"vertical_speed={state.vertical_speed:.2f} | "
            f"power_available={state.power_available:.2f} | "
            f"power_required={state.power_required_total:.2f} | "
            f"excess_power={state.excess_power:.2f} | "
            f"sep={state.sep:.2f} | "
            f"Vs1g={state.V_stall_1g:.2f} | "
            f"Vsm={state.V_stall_maneuver:.2f} | "
            f"margin1g={state.speed_margin_1g:.2f} | "
            f"marginM={state.speed_margin_maneuver:.2f} | "
            f"stall_warning={state.stall_warning} | "
            f"warn_margin={state.stall_warning_margin:.2f} | "
            f"n={state.load_factor:.2f} | "
            f"CLratio={state.CL_ratio:.2f} | "
            f"thrust={state.thrust:.1f} | "
            f"R={state.R:.3f}"
        )

        # защита от "разлёта"
        if state.V < 5:
            print("⚠ Скорость слишком мала (сваливание)")
            break

        if state.h < 0:
            print("⚠ Самолёт достиг земли")
            break

        # шаг времени
        t += dt
        if abs(state.theta) > np.radians(50) or abs(state.gamma) > np.radians(45):
            state.mode = "OUT_OF_ENVELOPE"
            print("⚠ Модель вышла за область применимости продольной 2D-модели")
            break
        visualizer.update(state, t)
        dtheta, dq, dV, dgamma, dh, dx, dsep, diag = dynamics.derivatives(state, u, throttle, dt)
        print(f"dtheta={dtheta:.6f}, dq={dq:.6f}, dV={dV:.6f}, dgamma={dgamma:.6f}")



# =========================
# ЗАПУСК
# =========================

if __name__ == "__main__":
    run_simulation()