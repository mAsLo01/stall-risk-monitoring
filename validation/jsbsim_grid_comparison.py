"""
validation/jsbsim_grid_comparison.py

Сравнение reduced-order модели с JSBSim по сетке режимов.

Проверяется не одна точка, а набор состояний:
- разные скорости V;
- разные углы атаки alpha;
- разные отклонения руля высоты delta_e.

Для каждой точки считаются:
- производные dV, dtheta, dgamma, dh, dq;
- эквивалентные аэродинамические коэффициенты CL, CD, Cm,
  восстановленные из JSBSim;
- ошибки собственной модели относительно JSBSim.

Назначение:
    python validation/jsbsim_grid_comparison.py

Результат:
    results/jsbsim_validation/grid_comparison.csv
    results/jsbsim_validation/grid_summary.csv
"""

import os
import sys
import csv
from dataclasses import dataclass

import numpy as np


# ============================================================
# Доступ к корневой папке проекта
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Импорты твоей модели
# ============================================================

from AircraftModel import BOEING_737_JSBSIM, CLEAN
from model.state import State
from model.dynamics import AircraftDynamics


# ============================================================
# JSBSim
# ============================================================

try:
    import jsbsim
except ImportError as exc:
    raise ImportError(
        "Не найден модуль jsbsim. Установи его через: pip install jsbsim"
    ) from exc


# ============================================================
# Настройки
# ============================================================

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "jsbsim_validation")

# Если у тебя JSBSim работает без root_dir, оставь None.
# Если нужен путь к aircraft/models, укажи его явно.
JSBSIM_ROOT = r"D:\учебка\вкр\VKR\jsbsim"

# Проверь имя модели: например "737", "c172x", "B737" — зависит от твоей установки.
JSBSIM_MODEL_NAME = "737"

# Какая reduced-order модель сравнивается
OWN_AIRCRAFT = BOEING_737_JSBSIM
OWN_CONFIG = CLEAN

# Малый шаг для численной оценки производных JSBSim
DT_COMPARE = 0.001

# Высота сравнения
H0_M = 500.0

# Начальный угол траектории.
# Для планирования ты уже использовала около -3.92 deg.
GAMMA0_DEG = -3.92

# Газ. Для чистого сравнения аэродинамики лучше 0.
THROTTLE_CMD = 0.0

# Сетка режимов.
V_GRID_MPS = [80.0, 120.0, 160.0, 200.0, 230.0]
ALPHA_GRID_DEG = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
DELTA_E_GRID_DEG = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]


# ============================================================
# Единицы
# ============================================================

FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M

FPS_TO_MPS = 0.3048
MPS_TO_FPS = 1.0 / FPS_TO_MPS


# ============================================================
# Утилиты JSBSim
# ============================================================

def make_jsbsim_exec(dt: float):
    """
    Создаёт и инициализирует объект JSBSim.
    """

    if JSBSIM_ROOT is None:
        fdm = jsbsim.FGFDMExec()
    else:
        fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)

    loaded = fdm.load_model(JSBSIM_MODEL_NAME)

    if not loaded:
        raise RuntimeError(f"JSBSim не смог загрузить модель: {JSBSIM_MODEL_NAME}")

    fdm.set_dt(dt)

    return fdm


def set_prop_safe(fdm, name: str, value: float):
    """
    Безопасная запись свойства JSBSim.

    JSBSim иногда молча принимает свойства, иногда нет —
    поэтому оставляем функцию отдельной, чтобы проще было отлаживать.
    """
    try:
        fdm.set_property_value(name, value)
        return True
    except Exception:
        return False


def get_prop_any(fdm, names: list[str], default=None):
    """
    Читает первое доступное свойство из списка.

    Это нужно, потому что у разных моделей/версий JSBSim
    названия некоторых каналов могут отличаться.
    """
    for name in names:
        try:
            value = fdm.get_property_value(name)
            if value is not None:
                return value
        except Exception:
            pass

    if default is not None:
        return default

    raise KeyError(f"Не удалось прочитать ни одно свойство из списка: {names}")


def set_initial_conditions(
        fdm,
        V_mps: float,
        h_m: float,
        alpha_deg: float,
        gamma_deg: float,
):
    """
    Задаёт начальные условия JSBSim.

    Используем:
    theta = gamma + alpha

    Важно:
    alpha и gamma задаются согласованно, чтобы JSBSim и твоя модель
    стартовали из одинаковой кинематической точки.
    """

    theta_deg = gamma_deg + alpha_deg

    # Скорость
    set_prop_safe(fdm, "ic/vt-fps", V_mps * MPS_TO_FPS)

    # Высота
    set_prop_safe(fdm, "ic/h-sl-ft", h_m * M_TO_FT)

    # Углы
    set_prop_safe(fdm, "ic/alpha-deg", alpha_deg)
    set_prop_safe(fdm, "ic/gamma-deg", gamma_deg)
    set_prop_safe(fdm, "ic/theta-deg", theta_deg)

    # Убираем начальные угловые скорости
    set_prop_safe(fdm, "ic/p-rad_sec", 0.0)
    set_prop_safe(fdm, "ic/q-rad_sec", 0.0)
    set_prop_safe(fdm, "ic/r-rad_sec", 0.0)

    # Убираем крен и рыскание для продольного сравнения
    set_prop_safe(fdm, "ic/phi-deg", 0.0)
    set_prop_safe(fdm, "ic/psi-true-deg", 0.0)

    fdm.run_ic()


def set_controls(
        fdm,
        delta_e_rad: float,
        throttle_cmd: float,
        aircraft=OWN_AIRCRAFT,
):
    """
    Задаёт управление в JSBSim.

    Здесь мы задаём и нормированную команду, и физическое положение руля.
    Это сделано специально для статического сравнения, чтобы не мешали
    динамика сервопривода/фильтры FCS.

    Если позже захочешь сравнивать именно работу FCS, прямую запись
    elevator-pos-rad можно убрать.
    """

    elevator_cmd_norm = delta_e_rad / aircraft.max_elevator
    elevator_cmd_norm = float(np.clip(elevator_cmd_norm, -1.0, 1.0))

    set_prop_safe(fdm, "fcs/elevator-cmd-norm", elevator_cmd_norm)
    set_prop_safe(fdm, "fcs/elevator-pos-norm", elevator_cmd_norm)
    set_prop_safe(fdm, "fcs/elevator-pos-rad", delta_e_rad)

    # Газ для двухдвигательной модели
    throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    for idx in range(4):
        set_prop_safe(fdm, f"fcs/throttle-cmd-norm[{idx}]", throttle_cmd)
        set_prop_safe(fdm, f"fcs/throttle-pos-norm[{idx}]", throttle_cmd)


def read_jsbsim_state(fdm):
    """
    Считывает основные переменные JSBSim в СИ.

    gamma считаем как theta - alpha, чтобы сравнение было согласовано
    с твоей reduced-order моделью, где alpha = theta - gamma.
    """

    V_mps = get_prop_any(
        fdm,
        ["velocities/vt-fps", "velocities/vc-fps"],
    ) * FPS_TO_MPS

    theta_rad = get_prop_any(
        fdm,
        ["attitude/theta-rad"],
    )

    alpha_rad = get_prop_any(
        fdm,
        ["aero/alpha-rad", "velocities/alpha-rad"],
    )

    gamma_rad = theta_rad - alpha_rad

    h_m = get_prop_any(
        fdm,
        ["position/h-sl-ft"],
    ) * FT_TO_M

    q_rad_s = get_prop_any(
        fdm,
        ["velocities/q-rad_sec", "velocities/q-aero-rad_sec"],
    )

    return {
        "V": V_mps,
        "theta": theta_rad,
        "alpha": alpha_rad,
        "gamma": gamma_rad,
        "h": h_m,
        "q": q_rad_s,
    }


def jsbsim_finite_derivatives(
        V_mps: float,
        h_m: float,
        alpha_deg: float,
        gamma_deg: float,
        delta_e_deg: float,
        throttle_cmd: float,
        dt: float,
):
    """
    Запускает JSBSim и оценивает производные конечной разностью.
    Первый шаг после задания управления используется для применения FCS.
    """

    fdm = make_jsbsim_exec(dt)

    set_initial_conditions(
        fdm=fdm,
        V_mps=V_mps,
        h_m=h_m,
        alpha_deg=alpha_deg,
        gamma_deg=gamma_deg,
    )

    delta_e_rad = np.radians(delta_e_deg)

    set_controls(
        fdm=fdm,
        delta_e_rad=delta_e_rad,
        throttle_cmd=throttle_cmd,
    )

    # ВАЖНО: дать JSBSim применить FCS/положение руля
    fdm.run()

    # После этого считаем состояние s0 уже с применённым управлением
    s0 = read_jsbsim_state(fdm)

    set_controls(
        fdm=fdm,
        delta_e_rad=delta_e_rad,
        throttle_cmd=throttle_cmd,
    )

    fdm.run()

    s1 = read_jsbsim_state(fdm)

    deriv = {
        "dV": (s1["V"] - s0["V"]) / dt,
        "dtheta": (s1["theta"] - s0["theta"]) / dt,
        "dalpha": (s1["alpha"] - s0["alpha"]) / dt,
        "dgamma": (s1["gamma"] - s0["gamma"]) / dt,
        "dh": (s1["h"] - s0["h"]) / dt,
        "dq": (s1["q"] - s0["q"]) / dt,
    }

    return s0, s1, deriv


# ============================================================
# Reduced-order model
# ============================================================

def own_model_derivatives(
        V_mps: float,
        h_m: float,
        alpha_deg: float,
        gamma_deg: float,
        delta_e_deg: float,
        throttle_cmd: float,
):
    """
    Считает производные твоей модели в той же точке.
    """

    aircraft = OWN_AIRCRAFT
    config = OWN_CONFIG

    dynamics = AircraftDynamics(aircraft, config)

    state = State()

    gamma_rad = np.radians(gamma_deg)
    alpha_rad = np.radians(alpha_deg)
    theta_rad = gamma_rad + alpha_rad

    state.V = V_mps
    state.h = h_m
    state.gamma = gamma_rad
    state.theta = theta_rad
    state.q = 0.0
    state.x = 0.0
    state.sep = 0.0

    delta_e_rad = np.radians(delta_e_deg)
    u = delta_e_rad / aircraft.max_elevator
    u = float(np.clip(u, -1.0, 1.0))

    diagnostics = dynamics.compute_diagnostics(
        state=state,
        u=u,
        throttle=throttle_cmd,
        dt=DT_COMPARE,
    )

    aero, forces, stall, moment, energy = diagnostics

    result = dynamics.derivatives(
        state=state,
        u=u,
        throttle=throttle_cmd,
        dt=DT_COMPARE,
    )

    dtheta, dq, dV, dgamma, dh, dx, dsep = result[:7]

    own = {
        "dV": dV,
        "dtheta": dtheta,
        "dgamma": dgamma,
        "dh": dh,
        "dq": dq,
        "dsep": dsep,

        "CL": aero.CL,
        "CD": aero.CD,
        "Cm": moment.Cm,

        "L": forces.L,
        "D": forces.D,
        "M": moment.M,

        "alpha": aero.alpha,
        "gamma": state.gamma,
        "theta": state.theta,
        "q": state.q,
        "rho": aero.rho,
        "u": u,
    }

    return own, state, dynamics


# ============================================================
# Восстановление коэффициентов JSBSim
# ============================================================

def reconstruct_jsbsim_coefficients(
        jsb_deriv: dict,
        jsb_state: dict,
        own_dynamics: AircraftDynamics,
        throttle_cmd: float,
):
    """
    Восстанавливает эквивалентные CL, CD, Cm из производных JSBSim
    через уравнения reduced-order модели.

    Это не прямые коэффициенты из внутренних таблиц JSBSim, а коэффициенты,
    которые JSBSim "эквивалентно показывает" в твоей системе уравнений.
    """

    aircraft = own_dynamics.aircraft
    g = own_dynamics.g

    V = max(jsb_state["V"], 1e-6)
    h = jsb_state["h"]
    alpha = jsb_state["alpha"]
    gamma = jsb_state["gamma"]

    rho = own_dynamics.atmosphere.density(h)
    q_dyn = 0.5 * rho * V ** 2

    T = own_dynamics.compute_thrust(throttle_cmd, V, alpha)
    T_x = T * np.cos(alpha)
    T_z = T * np.sin(alpha)

    W = aircraft.mass * g

    # Из уравнения:
    # dV = (T_x - D)/m - g sin(gamma)
    D_required = T_x - aircraft.mass * (
        jsb_deriv["dV"] + g * np.sin(gamma)
    )

    # Из уравнения:
    # dgamma = (Z - W cos(gamma))/(m V)
    # Z = L + T_z
    Z_required = (
        jsb_deriv["dgamma"] * aircraft.mass * V
        + W * np.cos(gamma)
    )

    L_required = Z_required - T_z

    # Из уравнения:
    # dq = M / Iyy
    M_required = jsb_deriv["dq"] * aircraft.Iyy

    CL_required = L_required / max(q_dyn * aircraft.wing_area, 1e-9)
    CD_required = D_required / max(q_dyn * aircraft.wing_area, 1e-9)
    Cm_required = M_required / max(
        q_dyn * aircraft.wing_area * aircraft.mean_chord,
        1e-9,
    )

    return {
        "CL": CL_required,
        "CD": CD_required,
        "Cm": Cm_required,
        "L": L_required,
        "D": D_required,
        "M": M_required,
        "q_dyn": q_dyn,
        "rho": rho,
    }


# ============================================================
# Одна точка сетки
# ============================================================

def compare_one_grid_point(
        V_mps: float,
        alpha_deg: float,
        delta_e_deg: float,
        gamma_deg: float = GAMMA0_DEG,
        h_m: float = H0_M,
        throttle_cmd: float = THROTTLE_CMD,
        dt: float = DT_COMPARE,
):
    """
    Сравнивает одну точку сетки.
    """

    jsb_s0, jsb_s1, jsb_deriv = jsbsim_finite_derivatives(
        V_mps=V_mps,
        h_m=h_m,
        alpha_deg=alpha_deg,
        gamma_deg=gamma_deg,
        delta_e_deg=delta_e_deg,
        throttle_cmd=throttle_cmd,
        dt=dt,
    )

    own_deriv, own_state, own_dynamics = own_model_derivatives(
        V_mps=V_mps,
        h_m=h_m,
        alpha_deg=alpha_deg,
        gamma_deg=gamma_deg,
        delta_e_deg=delta_e_deg,
        throttle_cmd=throttle_cmd,
    )

    jsb_coeff = reconstruct_jsbsim_coefficients(
        jsb_deriv=jsb_deriv,
        jsb_state=jsb_s0,
        own_dynamics=own_dynamics,
        throttle_cmd=throttle_cmd,
    )

    row = {
        "V_mps": V_mps,
        "h_m": h_m,
        "gamma_deg": gamma_deg,
        "alpha_deg": alpha_deg,
        "delta_e_deg": delta_e_deg,
        "throttle_cmd": throttle_cmd,

        "jsb_V": jsb_s0["V"],
        "jsb_theta_deg": np.degrees(jsb_s0["theta"]),
        "jsb_gamma_deg": np.degrees(jsb_s0["gamma"]),
        "jsb_alpha_deg": np.degrees(jsb_s0["alpha"]),
        "jsb_q": jsb_s0["q"],

        "own_dV": own_deriv["dV"],
        "jsb_dV": jsb_deriv["dV"],
        "err_dV": own_deriv["dV"] - jsb_deriv["dV"],

        "own_dtheta": own_deriv["dtheta"],
        "jsb_dtheta": jsb_deriv["dtheta"],
        "err_dtheta": own_deriv["dtheta"] - jsb_deriv["dtheta"],

        "own_dgamma": own_deriv["dgamma"],
        "jsb_dgamma": jsb_deriv["dgamma"],
        "err_dgamma": own_deriv["dgamma"] - jsb_deriv["dgamma"],

        "own_dh": own_deriv["dh"],
        "jsb_dh": jsb_deriv["dh"],
        "err_dh": own_deriv["dh"] - jsb_deriv["dh"],

        "own_dq": own_deriv["dq"],
        "jsb_dq": jsb_deriv["dq"],
        "err_dq": own_deriv["dq"] - jsb_deriv["dq"],

        "own_CL": own_deriv["CL"],
        "jsb_CL_eq": jsb_coeff["CL"],
        "err_CL": own_deriv["CL"] - jsb_coeff["CL"],

        "own_CD": own_deriv["CD"],
        "jsb_CD_eq": jsb_coeff["CD"],
        "err_CD": own_deriv["CD"] - jsb_coeff["CD"],

        "own_Cm": own_deriv["Cm"],
        "jsb_Cm_eq": jsb_coeff["Cm"],
        "err_Cm": own_deriv["Cm"] - jsb_coeff["Cm"],

        "jsb_q_dyn": jsb_coeff["q_dyn"],
        "jsb_rho": jsb_coeff["rho"],
    }

    return row


# ============================================================
# Сетка
# ============================================================

def run_grid_comparison():
    """
    Запускает сравнение по полной сетке.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    total = len(V_GRID_MPS) * len(ALPHA_GRID_DEG) * len(DELTA_E_GRID_DEG)
    counter = 0

    print("\n=== JSBSim grid comparison ===")
    print(f"Total points: {total}")
    print(f"Aircraft: {OWN_AIRCRAFT.name}")
    print(f"JSBSim model: {JSBSIM_MODEL_NAME}")
    print()

    for V_mps in V_GRID_MPS:
        for alpha_deg in ALPHA_GRID_DEG:
            for delta_e_deg in DELTA_E_GRID_DEG:
                counter += 1

                print(
                    f"[{counter:03d}/{total:03d}] "
                    f"V={V_mps:.1f} m/s | "
                    f"alpha={alpha_deg:+.1f} deg | "
                    f"delta_e={delta_e_deg:+.1f} deg"
                )

                try:
                    row = compare_one_grid_point(
                        V_mps=V_mps,
                        alpha_deg=alpha_deg,
                        delta_e_deg=delta_e_deg,
                    )
                    row["status"] = "OK"

                except Exception as exc:
                    row = {
                        "V_mps": V_mps,
                        "h_m": H0_M,
                        "gamma_deg": GAMMA0_DEG,
                        "alpha_deg": alpha_deg,
                        "delta_e_deg": delta_e_deg,
                        "throttle_cmd": THROTTLE_CMD,
                        "status": "ERROR",
                        "error_message": str(exc),
                    }
                    print(f"  ERROR: {exc}")

                rows.append(row)

    grid_path = os.path.join(RESULTS_DIR, "grid_comparison.csv")
    save_rows_csv(grid_path, rows)

    summary_rows = build_summary(rows)
    summary_path = os.path.join(RESULTS_DIR, "grid_summary.csv")
    save_rows_csv(summary_path, summary_rows)

    print()
    print("=== Grid comparison saved ===")
    print(f"Full table: {grid_path}")
    print(f"Summary:    {summary_path}")

    print_summary(summary_rows)

    return rows, summary_rows


# ============================================================
# Сохранение и summary
# ============================================================

def save_rows_csv(path: str, rows: list[dict]):
    if not rows:
        return

    fieldnames = sorted(set().union(*(row.keys() for row in rows)))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]):
    """
    Строит сводку по ошибкам.
    """

    ok_rows = [row for row in rows if row.get("status") == "OK"]

    if not ok_rows:
        return []

    error_columns = [
        "err_dV",
        "err_dtheta",
        "err_dgamma",
        "err_dh",
        "err_dq",
        "err_CL",
        "err_CD",
        "err_Cm",
    ]

    summary = []

    for col in error_columns:
        values = np.array([row[col] for row in ok_rows], dtype=float)
        abs_values = np.abs(values)

        summary.append({
            "metric": col,
            "mean_error": float(np.mean(values)),
            "mean_abs_error": float(np.mean(abs_values)),
            "max_abs_error": float(np.max(abs_values)),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
        })

    return summary


def print_summary(summary_rows: list[dict]):
    print()
    print("=== Summary ===")
    print(
        f"{'metric':>12} | "
        f"{'mean':>12} | "
        f"{'mean abs':>12} | "
        f"{'max abs':>12} | "
        f"{'rmse':>12}"
    )
    print("-" * 72)

    for row in summary_rows:
        print(
            f"{row['metric']:>12} | "
            f"{row['mean_error']:>12.6f} | "
            f"{row['mean_abs_error']:>12.6f} | "
            f"{row['max_abs_error']:>12.6f} | "
            f"{row['rmse']:>12.6f}"
        )


if __name__ == "__main__":
    run_grid_comparison()