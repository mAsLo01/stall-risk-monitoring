# validation/jsbsim_time_response.py

"""
Сравнение временного отклика собственной reduced-order модели и JSBSim.

Файл запускает две модели из одинаковых начальных условий:
- собственную модель AircraftDynamics + RK2Integrator;
- JSBSim-модель 737.

Затем обе модели интегрируются в течение t_final секунд при одинаковом
фиксированном управлении: elevator_cmd_norm и throttle_cmd.

Результат:
- CSV с траекториями own/jsbsim/error;
- summary в консоли.

Важно:
этот файл предназначен не для подбора коэффициентов, а для проверки того,
насколько уже подобранная reduced-order модель совпадает с JSBSim на коротком
временном интервале.
"""

import os
import sys
import csv
from dataclasses import dataclass

import numpy as np
import jsbsim


# ============================================================
# Пути проекта
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Импорты собственной модели
# ============================================================

from AircraftModel import BOEING_737_JSBSIM, CLEAN
from model.state import State
from model.dynamics import AircraftDynamics
from model.risk import RiskModel
from simulation.rk2 import RK2Integrator


# ============================================================
# Настройки JSBSim
# ============================================================

JSBSIM_ROOT = os.path.join(PROJECT_ROOT, "jsbsim")
JSBSIM_MODEL_NAME = "737"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "jsbsim_validation")


# ============================================================
# Конвертации единиц
# ============================================================

FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M

RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0


def mps_to_fps(v_mps: float) -> float:
    return v_mps * M_TO_FT


def fps_to_mps(v_fps: float) -> float:
    return v_fps * FT_TO_M


def m_to_ft(h_m: float) -> float:
    return h_m * M_TO_FT


def ft_to_m(h_ft: float) -> float:
    return h_ft * FT_TO_M


# ============================================================
# Состояние для сравнения
# ============================================================

@dataclass
class InitialCondition:
    V_mps: float = 80.0
    h_m: float = 500.0
    alpha_deg: float = 0.1957
    gamma_deg: float = -3.9257
    q_rad_s: float = 0.0

    @property
    def theta_deg(self) -> float:
        return self.gamma_deg + self.alpha_deg


# ============================================================
# Безопасная работа с JSBSim properties
# ============================================================

def get_prop(fdm, name: str, default=None):
    try:
        value = fdm.get_property_value(name)
        if value is None:
            return default
        return value
    except Exception:
        return default


def set_prop(fdm, name: str, value: float):
    try:
        fdm.set_property_value(name, value)
        return True
    except Exception:
        return False


def set_first_existing_prop(fdm, names, value: float):
    """
    Пробует установить первое существующее JSBSim-свойство из списка.
    Возвращает имя свойства, которое удалось установить.
    """
    for name in names:
        if set_prop(fdm, name, value):
            return name
    return None


# ============================================================
# JSBSim initialization
# ============================================================

def make_jsbsim_exec(dt: float):
    if not os.path.isdir(JSBSIM_ROOT):
        raise FileNotFoundError(
            f"Не найдена папка JSBSim data:\n{JSBSIM_ROOT}"
        )

    fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
    fdm.set_dt(dt)

    loaded = fdm.load_model(JSBSIM_MODEL_NAME)

    if not loaded:
        aircraft_dir = os.path.join(JSBSIM_ROOT, "aircraft")
        available = []

        if os.path.isdir(aircraft_dir):
            available = [
                name for name in os.listdir(aircraft_dir)
                if os.path.isdir(os.path.join(aircraft_dir, name))
            ]

        raise RuntimeError(
            f"JSBSim не смог загрузить модель: {JSBSIM_MODEL_NAME}\n"
            f"JSBSIM_ROOT = {JSBSIM_ROOT}\n"
            f"aircraft_dir = {aircraft_dir}\n"
            f"available aircraft folders = {available[:40]}"
        )

    return fdm


def set_jsbsim_initial_conditions(fdm, ic: InitialCondition):
    """
    Задаёт начальные условия JSBSim.

    Здесь theta задаётся явно как:
        theta = gamma + alpha

    Это соответствует твоей reduced-order модели:
        alpha = theta - gamma
    """

    theta_deg = ic.theta_deg

    # Основные initial condition properties
    set_prop(fdm, "ic/vt-fps", mps_to_fps(ic.V_mps))
    set_prop(fdm, "ic/h-sl-ft", m_to_ft(ic.h_m))

    set_prop(fdm, "ic/alpha-deg", ic.alpha_deg)
    set_prop(fdm, "ic/gamma-deg", ic.gamma_deg)
    set_prop(fdm, "ic/theta-deg", theta_deg)

    set_prop(fdm, "ic/q-rad_sec", ic.q_rad_s)

    # Нулевые боковые/угловые условия, чтобы сравнение было ближе к 2D
    set_prop(fdm, "ic/beta-deg", 0.0)
    set_prop(fdm, "ic/phi-deg", 0.0)
    set_prop(fdm, "ic/psi-deg", 0.0)

    set_prop(fdm, "ic/p-rad_sec", 0.0)
    set_prop(fdm, "ic/r-rad_sec", 0.0)

    ok = fdm.run_ic()

    if not ok:
        raise RuntimeError("JSBSim не смог выполнить run_ic().")


def set_jsbsim_controls(
        fdm,
        elevator_cmd_norm: float,
        throttle_cmd: float,
):
    """
    Устанавливает управление в JSBSim.

    elevator_cmd_norm:
        нормированная команда руля высоты, например -0.315.

    throttle_cmd:
        нормированная команда газа 0..1.
    """

    elevator_cmd_norm = float(np.clip(elevator_cmd_norm, -1.0, 1.0))
    throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    # Elevator
    elevator_prop = set_first_existing_prop(
        fdm,
        [
            "fcs/elevator-cmd-norm",
            "fcs/elevator-pos-norm",
        ],
        elevator_cmd_norm,
    )

    if elevator_prop is None:
        raise RuntimeError(
            "Не удалось задать elevator в JSBSim. "
            "Проверь названия FCS properties для модели."
        )

    # Throttle: для 737 обычно два двигателя
    set_prop(fdm, "fcs/throttle-cmd-norm", throttle_cmd)
    set_prop(fdm, "fcs/throttle-cmd-norm[0]", throttle_cmd)
    set_prop(fdm, "fcs/throttle-cmd-norm[1]", throttle_cmd)

    set_prop(fdm, "fcs/throttle-pos-norm", throttle_cmd)
    set_prop(fdm, "fcs/throttle-pos-norm[0]", throttle_cmd)
    set_prop(fdm, "fcs/throttle-pos-norm[1]", throttle_cmd)


def read_jsbsim_state(fdm):
    """
    Считывает состояние JSBSim в СИ.

    gamma напрямую в JSBSim может быть доступна не всегда одинаково.
    Для устойчивости считаем:
        gamma = theta - alpha

    Это делает сравнение согласованным с собственной моделью.
    """

    V = fps_to_mps(get_prop(fdm, "velocities/vt-fps", 0.0))
    h = ft_to_m(get_prop(fdm, "position/h-sl-ft", 0.0))

    theta = get_prop(fdm, "attitude/theta-rad", 0.0)
    alpha = get_prop(fdm, "aero/alpha-rad", 0.0)
    q = get_prop(fdm, "velocities/q-rad_sec", 0.0)

    gamma = theta - alpha

    return {
        "V": V,
        "h": h,
        "theta": theta,
        "alpha": alpha,
        "gamma": gamma,
        "q": q,
    }


# ============================================================
# Own model initialization
# ============================================================

def make_own_state(ic: InitialCondition) -> State:
    state = State()

    state.V = ic.V_mps
    state.h = ic.h_m
    state.gamma = np.radians(ic.gamma_deg)
    state.theta = np.radians(ic.theta_deg)
    state.alpha = np.radians(ic.alpha_deg)
    state.alpha_prev = state.alpha
    state.q = ic.q_rad_s

    state.x = 0.0
    state.sep = 0.0
    state.R = 0.0

    return state


def jsbsim_elevator_norm_to_own_u(elevator_cmd_norm: float) -> float:
    """
    Преобразование команды JSBSim в команду собственной модели.

    Сейчас предполагаем, что:
        u = elevator_cmd_norm

    Но если в твоей модели знак или масштаб отличаются, это единственное место,
    где нужно поменять соответствие.
    """

    return float(np.clip(elevator_cmd_norm, -1.0, 1.0))


# ============================================================
# Основное сравнение
# ============================================================

def compare_time_response_5s(
        t_final: float = 5.0,
        dt_local: float = 0.001,
        elevator_cmd_norm: float = -0.315,
        throttle_cmd: float = 0.0,
        initial_condition: InitialCondition | None = None,
        results_path: str | None = None,
):
    """
    Сравнивает временной отклик собственной модели и JSBSim.

    Возвращает список строк, также сохраняет CSV.
    """

    if initial_condition is None:
        initial_condition = InitialCondition()

    if results_path is None:
        results_path = os.path.join(
            RESULTS_DIR,
            "time_response_5s.csv",
        )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # -------------------------
    # JSBSim
    # -------------------------
    fdm = make_jsbsim_exec(dt_local)

    set_jsbsim_initial_conditions(fdm, initial_condition)
    set_jsbsim_controls(fdm, elevator_cmd_norm, throttle_cmd)

    # Один предварительный шаг нужен, чтобы JSBSim применил FCS/рули.
    fdm.run()
    set_jsbsim_controls(fdm, elevator_cmd_norm, throttle_cmd)

    jsb_initial_state = read_jsbsim_state(fdm)

    aircraft = BOEING_737_JSBSIM
    config = CLEAN

    dynamics = AircraftDynamics(aircraft, config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    own_state = State()
    own_state.V = jsb_initial_state["V"]
    own_state.h = jsb_initial_state["h"]
    own_state.theta = jsb_initial_state["theta"]
    own_state.gamma = jsb_initial_state["gamma"]
    own_state.alpha = jsb_initial_state["alpha"]
    own_state.alpha_prev = jsb_initial_state["alpha"]
    own_state.q = jsb_initial_state["q"]
    own_state.x = 0.0
    own_state.sep = 0.0
    own_state.R = 0.0

    own_u = jsbsim_elevator_norm_to_own_u(elevator_cmd_norm)

    # -------------------------
    # Time loop
    # -------------------------
    rows = []

    n_steps = int(np.round(t_final / dt_local))

    for i in range(n_steps + 1):
        t = i * dt_local

        jsb_state = read_jsbsim_state(fdm)

        row = make_comparison_row(
            t=t,
            own_state=own_state,
            jsb_state=jsb_state,
            elevator_cmd_norm=elevator_cmd_norm,
            own_u=own_u,
            throttle_cmd=throttle_cmd,
        )

        rows.append(row)

        if i == n_steps:
            break

        # Step own model
        own_state = integrator.step(
            state=own_state,
            dynamics=dynamics,
            risk_model=risk_model,
            u=own_u,
            throttle=throttle_cmd,
            dt=dt_local,
        )

        # Step JSBSim
        set_jsbsim_controls(fdm, elevator_cmd_norm, throttle_cmd)
        fdm.run()

    save_csv(results_path, rows)
    print_time_response_summary(rows, results_path)

    return rows


def make_comparison_row(
        t: float,
        own_state: State,
        jsb_state: dict,
        elevator_cmd_norm: float,
        own_u: float,
        throttle_cmd: float,
):
    own_alpha = own_state.theta - own_state.gamma

    row = {
        "t": t,

        "elevator_cmd_norm": elevator_cmd_norm,
        "own_u": own_u,
        "throttle_cmd": throttle_cmd,

        "own_V": own_state.V,
        "jsb_V": jsb_state["V"],
        "err_V": own_state.V - jsb_state["V"],

        "own_h": own_state.h,
        "jsb_h": jsb_state["h"],
        "err_h": own_state.h - jsb_state["h"],

        "own_theta_rad": own_state.theta,
        "jsb_theta_rad": jsb_state["theta"],
        "err_theta_rad": own_state.theta - jsb_state["theta"],
        "err_theta_deg": np.degrees(own_state.theta - jsb_state["theta"]),

        "own_gamma_rad": own_state.gamma,
        "jsb_gamma_rad": jsb_state["gamma"],
        "err_gamma_rad": own_state.gamma - jsb_state["gamma"],
        "err_gamma_deg": np.degrees(own_state.gamma - jsb_state["gamma"]),

        "own_alpha_rad": own_alpha,
        "jsb_alpha_rad": jsb_state["alpha"],
        "err_alpha_rad": own_alpha - jsb_state["alpha"],
        "err_alpha_deg": np.degrees(own_alpha - jsb_state["alpha"]),

        "own_q": own_state.q,
        "jsb_q": jsb_state["q"],
        "err_q": own_state.q - jsb_state["q"],

        "own_CL": getattr(own_state, "CL", 0.0),
        "own_CD": getattr(own_state, "CD", 0.0),
        "own_Cm": getattr(own_state, "Cm", 0.0),

        "own_sep": getattr(own_state, "sep", 0.0),
        "own_R": getattr(own_state, "R", 0.0),
        "own_mode": getattr(own_state, "mode", "UNKNOWN"),
    }

    return row


def save_csv(path: str, rows: list[dict]):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_time_response_summary(rows: list[dict], results_path: str):
    if not rows:
        print("No rows.")
        return

    max_abs_V = max(abs(r["err_V"]) for r in rows)
    max_abs_h = max(abs(r["err_h"]) for r in rows)
    max_abs_theta = max(abs(r["err_theta_deg"]) for r in rows)
    max_abs_gamma = max(abs(r["err_gamma_deg"]) for r in rows)
    max_abs_alpha = max(abs(r["err_alpha_deg"]) for r in rows)
    max_abs_q = max(abs(r["err_q"]) for r in rows)

    final = rows[-1]

    print("\n=== 5-second time response summary ===")
    print(f"max |V_err|       = {max_abs_V:.6f} m/s")
    print(f"max |theta_err|   = {max_abs_theta:.6f} deg")
    print(f"max |gamma_err|   = {max_abs_gamma:.6f} deg")
    print(f"max |alpha_err|   = {max_abs_alpha:.6f} deg")
    print(f"max |q_err|       = {max_abs_q:.6f} rad/s")
    print(f"max |h_err|       = {max_abs_h:.6f} m")

    print("\n=== Final state difference ===")
    print(f"t                = {final['t']:.3f} s")
    print(f"V_own - V_jsb    = {final['err_V']:.6f} m/s")
    print(f"theta error      = {final['err_theta_deg']:.6f} deg")
    print(f"gamma error      = {final['err_gamma_deg']:.6f} deg")
    print(f"alpha error      = {final['err_alpha_deg']:.6f} deg")
    print(f"q error          = {final['err_q']:.6f} rad/s")
    print(f"h error          = {final['err_h']:.6f} m")

    print(f"\nSaved CSV to: {results_path}")


# ============================================================
# Запуск
# ============================================================

def main():
    ic = InitialCondition(
        V_mps=80.0,
        h_m=500.0,
        alpha_deg=0.1957,
        gamma_deg=-3.9257,
        q_rad_s=0.0,
    )

    compare_time_response_5s(
        t_final=5.0,
        dt_local=0.001,
        elevator_cmd_norm=-0.315,
        throttle_cmd=0.0,
        initial_condition=ic,
        results_path=os.path.join(
            RESULTS_DIR,
            "time_response_5s.csv",
        ),
    )


if __name__ == "__main__":
    main()