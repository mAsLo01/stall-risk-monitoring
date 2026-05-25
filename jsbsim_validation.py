import csv
import numpy as np
import matplotlib.pyplot as plt
from jsbsim import FGFDMExec
import os
import sys

# Добавляем путь к вашим модулям, если скрипт не в корне проекта
sys.path.append(os.path.dirname(__file__))

from AircraftModel import BOEING_737_JSBSIM, CLEAN
from model.dynamics import AircraftDynamics
from model.risk import RiskModel
from simulation.rk2 import RK2Integrator
from copy import deepcopy
from dataclasses import dataclass
from config import dt
from model.state import State
from experiments.scenarios_737 import SCENARIOS_737 as SCENARIOS

LBS_TO_N = 4.4482216152605


@dataclass
class PitchMomentFitResult:
    Cm0: float
    Cm_alpha: float
    Cm_delta_e: float
    mean_abs_error: float
    max_abs_error: float

JSBSIM_DATA_PATH = os.path.join(os.getcwd(), "jsbsim")

# --- Параметры установившегося планирования (из аналитического теста) ---
V0 = 80.0          # м/с
alpha0 = 0.19     # градусов
gamma0 = -3.92     # градусов
theta0 = alpha0 + gamma0  # тангаж
h0 = 500.0         # начальная высота, м
u_elev = -0.315    # нормированное отклонение руля (из теста)
throttle = 0.0     # газ выключен
t_final = 30.0
M_TO_FT = 3.280839895
FT_TO_M = 0.3048
MPS_TO_FPS = 3.280839895
FPS_TO_MPS = 0.3048


def get_scenario_by_name(name: str):
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario

    available = [s.name for s in SCENARIOS]
    raise ValueError(
        f"Scenario '{name}' not found. Available scenarios: {available}"
    )



def replay_jsbsim_total_thrust_N(t, ref_result):
    thrust_0 = np.interp(t, ref_result["t"], ref_result["thrust_0_lbs"])
    thrust_1 = np.interp(t, ref_result["t"], ref_result["thrust_1_lbs"])
    return float((thrust_0 + thrust_1) * LBS_TO_N)


def throttle_for_target_thrust(dynamics, state, target_thrust_N):
    """
    Подбирает throttle для собственной модели так, чтобы compute_thrust()
    приблизительно совпал с фактической тягой JSBSim.
    """

    V = max(state.V, 1e-6)
    alpha = dynamics.compute_alpha(state.theta, state.gamma)

    lo = 0.0
    hi = 1.0

    for _ in range(30):
        mid = 0.5 * (lo + hi)
        T_mid = dynamics.compute_thrust(mid, V, alpha)

        if T_mid < target_thrust_N:
            lo = mid
        else:
            hi = mid

    return float(np.clip(0.5 * (lo + hi), 0.0, 1.0))


def set_jsbsim_initial_from_state(jsb, state):
    """
    Задаёт начальные условия JSBSim из State собственной модели.

    В собственной модели:
        alpha = theta - gamma

    Поэтому для JSBSim задаём:
        theta, gamma, alpha
    согласованно.
    """

    alpha_rad = state.theta - state.gamma

    jsb.set_property_value("ic/h-sl-ft", state.h * M_TO_FT)
    jsb.set_property_value("ic/vt-fps", state.V * MPS_TO_FPS)

    jsb.set_property_value("ic/theta-deg", np.degrees(state.theta))
    jsb.set_property_value("ic/gamma-deg", np.degrees(state.gamma))
    jsb.set_property_value("ic/alpha-deg", np.degrees(alpha_rad))

    jsb.set_property_value("ic/q-rad_sec", state.q)

    jsb.set_property_value("ic/phi-deg", 0.0)
    jsb.set_property_value("ic/psi-deg", 0.0)
    jsb.set_property_value("ic/beta-deg", 0.0)

    jsb.set_property_value("ic/p-rad_sec", 0.0)
    jsb.set_property_value("ic/r-rad_sec", 0.0)

    jsb.set_property_value("ic/lat-gc-deg", 0.0)
    jsb.set_property_value("ic/long-gc-deg", 0.0)


def make_state_from_jsbsim_state(jsb_state):
    state = State()

    state.V = jsb_state["V"]
    state.theta = jsb_state["theta"]
    state.gamma = jsb_state["gamma"]
    state.alpha = jsb_state["alpha"]
    state.alpha_prev = jsb_state["alpha"]
    state.q = jsb_state["q"]
    state.h = jsb_state["h"]

    state.x = 0.0
    state.R = 0.0
    state.sep = 0.0

    return state


def apply_jsbsim_configuration(jsb, scenario):
    """
    Приводит конфигурацию JSBSim к CLEAN / LANDING.
    Названия свойств могут отличаться у конкретной модели 737,
    поэтому используем set_if_exists.
    """

    config_name = getattr(scenario.config, "name", "").lower()

    if "landing" in config_name:
        flap_cmd = 1.0
        gear_cmd = 1.0
    else:
        flap_cmd = 0.0
        gear_cmd = 0.0

    set_if_exists(jsb, "fcs/flap-cmd-norm", flap_cmd)
    set_if_exists(jsb, "fcs/flap-pos-norm", flap_cmd)
    set_if_exists(jsb, "gear/gear-cmd-norm", gear_cmd)
    set_if_exists(jsb, "gear/gear-pos-norm", gear_cmd)


def safe_get(fdm, name, default=np.nan):
    try:
        value = fdm.get_property_value(name)
        if value is None:
            return default
        return value
    except Exception:
        return default


def get_jsbsim_state(jsb):
    """
    Считывает основные каналы из JSBSim в SI.

    Важно:
    gamma лучше брать из JSBSim, если свойство есть.
    Если его нет, считаем gamma через vertical speed:
        gamma = asin(hdot / V)
    """

    V = safe_get(jsb, "velocities/vt-fps", 0.0) * FPS_TO_MPS

    theta = safe_get(jsb, "attitude/theta-rad", None)
    if theta is None:
        theta = safe_get(jsb, "attitude/pitch-rad", 0.0)

    q = safe_get(jsb, "velocities/q-rad_sec", 0.0)

    h = safe_get(jsb, "position/h-sl-ft", 0.0) * FT_TO_M

    alpha = safe_get(jsb, "aero/alpha-rad", None)

    gamma = safe_get(jsb, "flight-path/gamma-rad", None)



    if gamma is None:
        gamma = safe_get(jsb, "velocities/gamma-rad", None)

    if gamma is None:
        hdot_fps = safe_get(jsb, "velocities/h-dot-fps", 0.0)
        hdot = hdot_fps * FPS_TO_MPS

        if abs(V) > 1e-6:
            gamma = np.arcsin(np.clip(hdot / V, -1.0, 1.0))
        else:
            gamma = 0.0

    if alpha is None:
        alpha = theta - gamma

    return {
        "V": V,
        "theta": theta,
        "gamma": gamma,
        "alpha": alpha,
        "q": q,
        "h": h,
    }


def initialize_jsbsim_engines(jsb, throttle_cmd=0.3):
    """
    Инициализация обоих двигателей JSBSim.
    """

    # Не оставляем активным только один двигатель.
    # Некоторые модели JSBSim используют active_engine как выбранный двигатель
    # для последующих команд запуска.
    for engine_idx in [0, 1]:
        set_if_exists(jsb, "propulsion/active_engine", engine_idx)

        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/set-running", 1)
        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/running", 1)
        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/starter_cmd", 1)
        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/cutoff", 0)
        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/mixture-cmd-norm", 1.0)
        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/condition", 1.0)

        set_if_exists(jsb, f"fcs/throttle-cmd-norm[{engine_idx}]", throttle_cmd)
        set_if_exists(jsb, f"fcs/throttle-pos-norm[{engine_idx}]", throttle_cmd)
        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/throttle", throttle_cmd)
        set_if_exists(jsb, f"propulsion/engine[{engine_idx}]/throttle-cmd-norm", throttle_cmd)

    set_if_exists(jsb, "fcs/throttle-cmd-norm", throttle_cmd)
    set_if_exists(jsb, "fcs/throttle-pos-norm", throttle_cmd)

    for _ in range(100):
        for engine_idx in [0, 1]:
            set_if_exists(jsb, f"fcs/throttle-cmd-norm[{engine_idx}]", throttle_cmd)
            set_if_exists(jsb, f"fcs/throttle-pos-norm[{engine_idx}]", throttle_cmd)
        jsb.run()


def reset_jsbsim_to_scenario_initial_state(jsb, scenario):
    """
    Возвращает JSBSim в начальное состояние сценария после прогрева двигателей/FCS.
    Это нужно, потому что initialize_jsbsim_engines() делает jsb.run()
    и самолёт успевает изменить V, h, gamma, theta.
    """

    initial_state = scenario.make_initial_state()

    set_jsbsim_initial_from_state(jsb, initial_state)

    ok = jsb.run_ic()
    if not ok:
        raise RuntimeError("JSBSim run_ic() failed after scenario reset.")

    return initial_state


def run_jsbsim_scenario(scenario, rotation_mode: str):
    fdm = FGFDMExec(JSBSIM_DATA_PATH)
    fdm.load_model("737")
    fdm.set_dt(dt)

    if rotation_mode == "3dof":
        fdm.set_property_value("simulation/rotation", 0)
    else:
        fdm.set_property_value("simulation/rotation", 1)

    # 1. Берём исходное состояние сценария.
    initial_state = scenario.make_initial_state()

    # 2. Берём начальные команды из сценария.
    u0, throttle0 = scenario.control_law(0.0, initial_state)

    # 3. Сначала задаём IC и делаем run_ic.
    set_jsbsim_initial_from_state(fdm, initial_state)
    fdm.run_ic()

    # 4. Прогреваем двигатели.
    # ВАЖНО: эта функция делает fdm.run(), поэтому самолёт смещается.
    initialize_jsbsim_engines(fdm, throttle_cmd=throttle0)

    # 5. Применяем конфигурацию.
    apply_jsbsim_configuration(fdm, scenario)

    # 6. После прогрева ОБЯЗАТЕЛЬНО возвращаем самолёт
    # в исходное состояние сценария.
    initial_state = reset_jsbsim_to_scenario_initial_state(fdm, scenario)

    # 7. После reset ещё раз применяем конфигурацию и управление t=0.
    apply_jsbsim_configuration(fdm, scenario)

    u0, throttle0 = scenario.control_law(0.0, initial_state)

    apply_jsbsim_controls(
        fdm,
        elevator_cmd_norm=u0,
        throttle_cmd=throttle0
    )

    # 8. Теперь это реальная стартовая точка сравнения.
    jsb_initial_state = get_jsbsim_state(fdm)
    initial_state_for_own = make_state_from_jsbsim_state(jsb_initial_state)

    t_start = fdm.get_property_value("simulation/sim-time-sec")

    t_arr = []
    V_arr = []
    alpha_arr = []
    h_arr = []
    theta_arr = []
    gamma_arr = []
    q_arr = []
    u_arr = []
    throttle_arr = []

    throttle_pos_0_arr = []
    throttle_pos_1_arr = []
    thrust_0_lbs_arr = []
    thrust_1_lbs_arr = []
    elevator_pos_rad_arr = []
    flap_pos_arr = []
    gear_pos_arr = []

    # 9. Сохраняем строку t=0 ДО первого шага fdm.run().
    t_arr.append(0.0)
    V_arr.append(jsb_initial_state["V"])
    alpha_arr.append(np.degrees(jsb_initial_state["alpha"]))
    h_arr.append(jsb_initial_state["h"])
    theta_arr.append(np.degrees(jsb_initial_state["theta"]))
    gamma_arr.append(np.degrees(jsb_initial_state["gamma"]))
    q_arr.append(jsb_initial_state["q"])
    u_arr.append(u0)
    throttle_arr.append(throttle0)

    throttle_pos_0_arr.append(safe_get(fdm, "fcs/throttle-pos-norm[0]", np.nan))
    throttle_pos_1_arr.append(safe_get(fdm, "fcs/throttle-pos-norm[1]", np.nan))
    thrust_0_lbs_arr.append(safe_get(fdm, "propulsion/engine[0]/thrust-lbs", np.nan))
    thrust_1_lbs_arr.append(safe_get(fdm, "propulsion/engine[1]/thrust-lbs", np.nan))
    elevator_pos_rad_arr.append(safe_get(fdm, "fcs/elevator-pos-rad", np.nan))
    flap_pos_arr.append(safe_get(fdm, "fcs/flap-pos-norm", np.nan))
    gear_pos_arr.append(safe_get(fdm, "gear/gear-pos-norm", np.nan))

    while True:
        t_abs = fdm.get_property_value("simulation/sim-time-sec")
        t = t_abs - t_start

        jsb_state = get_jsbsim_state(fdm)
        control_state = make_state_from_jsbsim_state(jsb_state)

        u, thr = scenario.control_law(t, control_state)

        apply_jsbsim_controls(
            fdm,
            elevator_cmd_norm=u,
            throttle_cmd=thr
        )

        fdm.run()

        t_abs_new = fdm.get_property_value("simulation/sim-time-sec")
        t_new = t_abs_new - t_start

        jsb_state_new = get_jsbsim_state(fdm)

        t_arr.append(t_new)
        V_arr.append(jsb_state_new["V"])
        alpha_arr.append(np.degrees(jsb_state_new["alpha"]))
        h_arr.append(jsb_state_new["h"])
        theta_arr.append(np.degrees(jsb_state_new["theta"]))
        gamma_arr.append(np.degrees(jsb_state_new["gamma"]))
        q_arr.append(jsb_state_new["q"])
        u_arr.append(u)
        throttle_arr.append(thr)

        throttle_pos_0_arr.append(safe_get(fdm, "fcs/throttle-pos-norm[0]", np.nan))
        throttle_pos_1_arr.append(safe_get(fdm, "fcs/throttle-pos-norm[1]", np.nan))
        thrust_0_lbs_arr.append(safe_get(fdm, "propulsion/engine[0]/thrust-lbs", np.nan))
        thrust_1_lbs_arr.append(safe_get(fdm, "propulsion/engine[1]/thrust-lbs", np.nan))
        elevator_pos_rad_arr.append(safe_get(fdm, "fcs/elevator-pos-rad", np.nan))
        flap_pos_arr.append(safe_get(fdm, "fcs/flap-pos-norm", np.nan))
        gear_pos_arr.append(safe_get(fdm, "gear/gear-pos-norm", np.nan))

        if t_new >= scenario.t_final or jsb_state_new["h"] <= 0.0:
            break

    return {
        "t": np.array(t_arr),
        "V": np.array(V_arr),
        "alpha": np.array(alpha_arr),
        "h": np.array(h_arr),
        "theta": np.array(theta_arr),
        "gamma": np.array(gamma_arr),
        "q": np.array(q_arr),
        "u": np.array(u_arr),
        "throttle": np.array(throttle_arr),

        "throttle_pos_0": np.array(throttle_pos_0_arr),
        "throttle_pos_1": np.array(throttle_pos_1_arr),
        "thrust_0_lbs": np.array(thrust_0_lbs_arr),
        "thrust_1_lbs": np.array(thrust_1_lbs_arr),
        "elevator_pos_rad": np.array(elevator_pos_rad_arr),
        "flap_pos": np.array(flap_pos_arr),
        "gear_pos": np.array(gear_pos_arr),

        "initial_state_for_own": initial_state_for_own,
    }


def make_state_from_jsbsim_result_initial(ref_result):
    """
    Создаёт State собственной модели из начального состояния JSBSim,
    сохранённого в ref_result.
    """

    return deepcopy(ref_result["initial_state_for_own"])


def replay_actual_elevator_from_jsbsim(t, ref_result, aircraft):
    """
    Возвращает u для собственной модели по фактическому положению руля JSBSim.

    JSBSim хранит elevator_pos_rad в радианах.
    Собственная модель ждёт нормированную команду u:
        delta = u * aircraft.max_elevator
    """

    elevator_rad = np.interp(
        t,
        ref_result["t"],
        ref_result["elevator_pos_rad"]
    )

    u = elevator_rad / aircraft.max_elevator

    return float(np.clip(u, -1.0, 1.0))


def run_own_model_scenario(scenario):
    """
    Запускает собственную reduced-order модель по объекту Scenario.
    """

    aircraft = BOEING_737_JSBSIM
    config = scenario.config

    dynamics = AircraftDynamics(aircraft, config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    state = scenario.make_initial_state()

    t_arr = []
    V_arr = []
    alpha_arr = []
    h_arr = []
    theta_arr = []
    gamma_arr = []
    q_arr = []
    u_arr = []
    throttle_arr = []
    R_arr = []
    sep_arr = []
    mode_arr = []

    t = 0.0

    while t <= scenario.t_final:
        u, thr = scenario.control_law(t, state)

        state = integrator.step(
            state,
            dynamics,
            risk_model,
            u,
            thr,
            dt
        )

        alpha = dynamics.compute_alpha(state.theta, state.gamma)

        t_arr.append(t)
        V_arr.append(state.V)
        alpha_arr.append(np.degrees(alpha))
        h_arr.append(state.h)
        theta_arr.append(np.degrees(state.theta))
        gamma_arr.append(np.degrees(state.gamma))
        q_arr.append(state.q)
        u_arr.append(u)
        throttle_arr.append(thr)
        R_arr.append(state.R)
        sep_arr.append(state.sep)
        mode_arr.append(state.mode)

        t += dt

        if state.h <= 0.0:
            print("Собственная модель достигла земли")
            break

        if state.V <= 5.0:
            print("Собственная модель вышла на слишком малую скорость")
            break

    return {
        "t": np.array(t_arr),
        "V": np.array(V_arr),
        "alpha": np.array(alpha_arr),
        "h": np.array(h_arr),
        "theta": np.array(theta_arr),
        "gamma": np.array(gamma_arr),
        "q": np.array(q_arr),
        "u": np.array(u_arr),
        "throttle": np.array(throttle_arr),
        "R": np.array(R_arr),
        "sep": np.array(sep_arr),
        "mode": np.array(mode_arr, dtype=object),

    }


def get_array_value(result, key, i, default=np.nan):
    if key not in result:
        return default

    arr = result[key]

    try:
        if len(arr) <= i:
            return default
        return arr[i]
    except TypeError:
        return arr

def add_dq_estimate_to_result(res):
    q = np.asarray(res["q"], dtype=float)
    t = np.asarray(res["t"], dtype=float)

    dq = np.full_like(q, np.nan)

    for i in range(1, len(q)):
        dt_i = t[i] - t[i - 1]
        if dt_i > 0:
            dq[i] = (q[i] - q[i - 1]) / dt_i

    res["dq_est"] = dq


def add_gamma_derivative_and_load_factor_estimate(res):
    gamma_deg = np.asarray(res["gamma"], dtype=float)
    gamma_rad = np.radians(gamma_deg)
    V = np.asarray(res["V"], dtype=float)
    t = np.asarray(res["t"], dtype=float)

    dgamma = np.full_like(gamma_rad, np.nan)
    load_factor_est = np.full_like(gamma_rad, np.nan)

    g = 9.81

    for i in range(1, len(gamma_rad)):
        dt_i = t[i] - t[i - 1]

        if dt_i > 0:
            dgamma[i] = (gamma_rad[i] - gamma_rad[i - 1]) / dt_i

            # Из уравнения:
            # dgamma = (Z - W*cos(gamma)) / (m*V)
            # n = Z / W = cos(gamma) + V/g * dgamma
            load_factor_est[i] = np.cos(gamma_rad[i]) + (V[i] / g) * dgamma[i]

    res["dgamma_est"] = dgamma
    res["load_factor_est"] = load_factor_est


def kcas_to_mps(kcas):
    return kcas * 0.514444


def first_time_event(t, values, condition):
    for ti, vi in zip(t, values):
        if np.isfinite(vi) and condition(vi):
            return float(ti)
    return None


def compute_event_metrics(result, scenario, model_name):
    t = np.asarray(result["t"], dtype=float)
    alpha = np.asarray(result["alpha"], dtype=float)
    h = np.asarray(result["h"], dtype=float)
    V = np.asarray(result["V"], dtype=float)

    speed_margin = compute_speed_margin(result, scenario)

    t_alpha_8 = first_time_event(t, alpha, lambda x: x >= 8.0)
    t_alpha_10 = first_time_event(t, alpha, lambda x: x >= 10.0)
    t_alpha_12 = first_time_event(t, alpha, lambda x: x >= 12.0)
    t_alpha_14 = first_time_event(t, alpha, lambda x: x >= 14.0)
    t_alpha_15 = first_time_event(t, alpha, lambda x: x >= 15.0)

    t_margin_110 = first_time_event(t, speed_margin, lambda x: x <= 1.10)
    t_margin_105 = first_time_event(t, speed_margin, lambda x: x <= 1.05)
    t_margin_100 = first_time_event(t, speed_margin, lambda x: x <= 1.00)

    if t_alpha_15 is not None and t_margin_100 is not None:
        stall_proxy_reason = "alpha_and_margin"
    elif t_alpha_15 is not None:
        stall_proxy_reason = "alpha"
    elif t_margin_100 is not None:
        stall_proxy_reason = "speed_margin"
    else:
        stall_proxy_reason = None

    # Унифицированный warning proxy:
    # либо alpha >= 12°, либо запас по скорости <= 1.10
    warning_candidates = [
        x for x in [t_alpha_12, t_margin_110]
        if x is not None
    ]
    t_warning_proxy = min(warning_candidates) if warning_candidates else None

    # Унифицированный stall proxy:
    # либо alpha >= 15°, либо speed margin <= 1.00
    stall_candidates = [
        x for x in [t_alpha_15, t_margin_100]
        if x is not None
    ]
    t_stall_proxy = min(stall_candidates) if stall_candidates else None

    # Для own дополнительно используем внутренние sep/mode, если они есть
    t_stall_internal = None
    t_warning_internal = None

    if "sep" in result:
        sep = np.asarray(result["sep"], dtype=float)
        t_sep_05 = first_time_event(t, sep, lambda x: x >= 0.5)
        t_sep_07 = first_time_event(t, sep, lambda x: x >= 0.7)
    else:
        sep = None
        t_sep_05 = None
        t_sep_07 = None

    if "mode" in result:
        mode = np.asarray(result["mode"], dtype=object)

        for ti, mi in zip(t, mode):
            if str(mi) in ["WARNING", "STALL"]:
                t_warning_internal = float(ti)
                break

        for ti, mi in zip(t, mode):
            if str(mi) == "STALL":
                t_stall_internal = float(ti)
                break
    else:
        mode = None

    if t_stall_internal is None and t_sep_07 is not None:
        t_stall_internal = t_sep_07

    h0 = h[0]
    max_h = np.maximum.accumulate(h)

    def altitude_loss_at(t_event):
        if t_event is None:
            return None, None

        idx = int(np.argmin(np.abs(t - t_event)))
        loss_from_start = max(0.0, h0 - h[idx])
        loss_from_peak = max(0.0, max_h[idx] - h[idx])

        return float(loss_from_start), float(loss_from_peak)

    loss_warning_start, loss_warning_peak = altitude_loss_at(t_warning_proxy)
    loss_stall_start, loss_stall_peak = altitude_loss_at(t_stall_proxy)

    if t_stall_proxy is not None:
        phase_reached = "stall_proxy"
    elif t_warning_proxy is not None:
        phase_reached = "warning_proxy"
    elif t_alpha_12 is not None:
        phase_reached = "alpha_12"
    elif t_alpha_10 is not None:
        phase_reached = "alpha_10"
    elif t_alpha_8 is not None:
        phase_reached = "alpha_8"
    else:
        phase_reached = "no_event"

    return {
        "scenario": scenario.name,
        "model": model_name,
        "phase_reached": phase_reached,
        "t_end": float(t[-1]) if len(t) > 0 else None,

        "t_alpha_8": t_alpha_8,
        "t_alpha_10": t_alpha_10,
        "t_alpha_12": t_alpha_12,
        "t_alpha_14": t_alpha_14,
        "t_alpha_15": t_alpha_15,

        "t_margin_110": t_margin_110,
        "t_margin_105": t_margin_105,
        "t_margin_100": t_margin_100,

        "t_warning_proxy": t_warning_proxy,
        "t_stall_proxy": t_stall_proxy,

        "t_warning_internal": t_warning_internal,
        "t_stall_internal": t_stall_internal,
        "t_sep_05": t_sep_05,
        "t_sep_07": t_sep_07,

        "max_alpha": float(np.nanmax(alpha)),
        "min_V": float(np.nanmin(V)),
        "min_speed_margin": float(np.nanmin(speed_margin)),
        "min_h": float(np.nanmin(h)),

        "altitude_loss_before_warning": loss_warning_start,
        "altitude_loss_from_peak_before_warning": loss_warning_peak,
        "altitude_loss_before_stall": loss_stall_start,
        "altitude_loss_from_peak_before_stall": loss_stall_peak,

        "stall_proxy_reason": stall_proxy_reason,
        "t_alpha_15": t_alpha_15,
        "t_margin_100": t_margin_100,
    }

def compute_event_error_row(event_rows, scenario):
    """
    Строит строку ошибок событий:
        error = own - jsbsim_6dof

    Если событие отсутствует у одной из моделей, ошибка по этому событию = None.
    """

    own = None
    jsb = None

    for row in event_rows:
        if row.get("model") == "own":
            own = row
        elif row.get("model") == "jsbsim_6dof":
            jsb = row

    if own is None or jsb is None:
        return {
            "scenario": scenario.name,
            "model": "own_minus_jsbsim_6dof",
            "comparison_valid": False,
            "error": "Missing own or jsbsim_6dof event row",
        }

    def diff(key):
        a = own.get(key)
        b = jsb.get(key)

        if a is None or b is None:
            return None

        try:
            if not np.isfinite(a) or not np.isfinite(b):
                return None
        except TypeError:
            return None

        return float(a - b)

    def absdiff(key):
        value = diff(key)
        return None if value is None else abs(value)

    event_keys = [
        "t_alpha_8",
        "t_alpha_10",
        "t_alpha_12",
        "t_alpha_14",
        "t_alpha_15",
        "t_margin_110",
        "t_margin_105",
        "t_margin_100",
        "t_warning_proxy",
        "t_stall_proxy",
        "max_alpha",
        "min_V",
        "min_speed_margin",
        "min_h",
        "altitude_loss_before_warning",
        "altitude_loss_from_peak_before_warning",
        "altitude_loss_before_stall",
        "altitude_loss_from_peak_before_stall",
    ]

    row = {
        "scenario": scenario.name,
        "model": "own_minus_jsbsim_6dof",
        "scenario_group": getattr(scenario, "group", ""),
        "scenario_purpose": getattr(scenario, "purpose", ""),
        "parent_trim": getattr(scenario, "parent_trim", ""),
        "primary_for_validation": getattr(scenario, "primary_for_validation", False),

        "comparison_valid": True,

        "own_phase_reached": own.get("phase_reached"),
        "jsb_phase_reached": jsb.get("phase_reached"),
        "same_phase_reached": own.get("phase_reached") == jsb.get("phase_reached"),

        "own_warning_reached": own.get("t_warning_proxy") is not None,
        "jsb_warning_reached": jsb.get("t_warning_proxy") is not None,
        "own_stall_reached": own.get("t_stall_proxy") is not None,
        "jsb_stall_reached": jsb.get("t_stall_proxy") is not None,
    }

    for key in event_keys:
        row[f"err_{key}"] = diff(key)
        row[f"abs_err_{key}"] = absdiff(key)

    # Удобные агрегаты для ранней фазы
    early_abs_errors = [
        row.get("abs_err_t_alpha_8"),
        row.get("abs_err_t_alpha_10"),
        row.get("abs_err_t_alpha_12"),
        row.get("abs_err_t_warning_proxy"),
    ]

    early_abs_errors = [
        x for x in early_abs_errors
        if x is not None and np.isfinite(x)
    ]

    if early_abs_errors:
        row["mean_abs_early_event_error_s"] = float(np.mean(early_abs_errors))
        row["max_abs_early_event_error_s"] = float(np.max(early_abs_errors))
    else:
        row["mean_abs_early_event_error_s"] = None
        row["max_abs_early_event_error_s"] = None

    return row

def compute_speed_margin(result, scenario):
    """
    Унифицированная оценка запаса по скорости для own и JSBSim.

    speed_margin = V / V_stall_maneuver
    V_stall_maneuver = V_stall_1g * sqrt(load_factor)

    Если load_factor_est отсутствует, используем n=1.
    """

    V = np.asarray(result["V"], dtype=float)

    if "load_factor_est" in result:
        n = np.asarray(result["load_factor_est"], dtype=float)
    elif "load_factor" in result:
        n = np.asarray(result["load_factor"], dtype=float)
    else:
        n = np.ones_like(V)

    n = np.where(np.isfinite(n), n, 1.0)
    n = np.maximum(n, 0.1)

    V_stall_1g = kcas_to_mps(scenario.config.stall_speed_kcas)
    V_stall_maneuver = V_stall_1g * np.sqrt(n)

    return V / np.maximum(V_stall_maneuver, 1e-9)


def save_event_metrics_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def print_event_comparison(rows):
    print("\n=== Event metrics comparison ===")

    for row in rows:
        print(
            f"{row['model']:10s} | "
            f"t_a10={row['t_alpha_10']} | "
            f"t_a12={row['t_alpha_12']} | "
            f"warn={row['t_warning_proxy']} | "
            f"stall={row['t_stall_proxy']} | "
            f"max_alpha={row['max_alpha']:.2f} | "
            f"min_margin={row['min_speed_margin']:.3f}"
        )


def save_scenario_comparison_csv(path, scenario_name, res_own, res_3dof, res_6dof):
    """
    Сохраняет сравнение трёх моделей в один CSV.
    Длины массивов могут чуть отличаться, поэтому сохраняем по минимальной длине.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    n = min(
        len(res_own["t"]),
        len(res_3dof["t"]),
        len(res_6dof["t"]),
    )

    fieldnames = [
        "scenario",
        "t",

        "own_V", "jsb3_V", "jsb6_V",
        "own_alpha", "jsb3_alpha", "jsb6_alpha",
        "own_h", "jsb3_h", "jsb6_h",
        "own_theta", "jsb3_theta", "jsb6_theta",
        "own_gamma", "jsb3_gamma", "jsb6_gamma",
        "own_q", "jsb3_q", "jsb6_q",

        "own_u", "jsb3_u", "jsb6_u",
        "own_throttle", "jsb3_throttle", "jsb6_throttle",

        "err_V_own_jsb6",
        "err_alpha_own_jsb6",
        "err_h_own_jsb6",
        "err_theta_own_jsb6",
        "err_gamma_own_jsb6",
        "err_q_own_jsb6",

        "jsb3_throttle_pos_0",
        "jsb3_throttle_pos_1",
        "jsb3_thrust_0_lbs",
        "jsb3_thrust_1_lbs",
        "jsb3_elevator_pos_rad",
        "jsb3_flap_pos",
        "jsb3_gear_pos",

        "jsb6_throttle_pos_0",
        "jsb6_throttle_pos_1",
        "jsb6_thrust_0_lbs",
        "jsb6_thrust_1_lbs",
        "jsb6_elevator_pos_rad",
        "jsb6_flap_pos",
        "jsb6_gear_pos",

        "own_thrust_N",
        "jsb6_total_thrust_N",

        "own_CL",
        "own_CD",
        "own_load_factor",
        "own_vertical_speed",

        "own_Cm",
        "own_delta_e_rad",
        "own_q_dyn",
        "own_dq_est",
        "jsb6_dq_est",

        "own_load_factor_est",
        "jsb6_load_factor_est",
        "own_dgamma_est",
        "jsb6_dgamma_est",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";"
        )
        writer.writeheader()

        for i in range(n):
            row = {
                "scenario": scenario_name,
                "t": res_own["t"][i],

                "own_V": res_own["V"][i],
                "jsb3_V": res_3dof["V"][i],
                "jsb6_V": res_6dof["V"][i],

                "own_alpha": res_own["alpha"][i],
                "jsb3_alpha": res_3dof["alpha"][i],
                "jsb6_alpha": res_6dof["alpha"][i],

                "own_h": res_own["h"][i],
                "jsb3_h": res_3dof["h"][i],
                "jsb6_h": res_6dof["h"][i],

                "own_theta": res_own["theta"][i],
                "jsb3_theta": res_3dof["theta"][i],
                "jsb6_theta": res_6dof["theta"][i],

                "own_gamma": res_own["gamma"][i],
                "jsb3_gamma": res_3dof["gamma"][i],
                "jsb6_gamma": res_6dof["gamma"][i],

                "own_q": res_own["q"][i],
                "jsb3_q": res_3dof["q"][i],
                "jsb6_q": res_6dof["q"][i],

                "own_u": res_own["u"][i],
                "jsb3_u": res_3dof["u"][i],
                "jsb6_u": res_6dof["u"][i],

                "own_throttle": res_own["throttle"][i],
                "jsb3_throttle": res_3dof["throttle"][i],
                "jsb6_throttle": res_6dof["throttle"][i],

                "err_V_own_jsb6": res_own["V"][i] - res_6dof["V"][i],
                "err_alpha_own_jsb6": res_own["alpha"][i] - res_6dof["alpha"][i],
                "err_h_own_jsb6": res_own["h"][i] - res_6dof["h"][i],
                "err_theta_own_jsb6": res_own["theta"][i] - res_6dof["theta"][i],
                "err_gamma_own_jsb6": res_own["gamma"][i] - res_6dof["gamma"][i],
                "err_q_own_jsb6": res_own["q"][i] - res_6dof["q"][i],
                "jsb3_throttle_pos_0": get_array_value(res_3dof, "throttle_pos_0", i),
                "jsb3_throttle_pos_1": get_array_value(res_3dof, "throttle_pos_1", i),
                "jsb3_thrust_0_lbs": get_array_value(res_3dof, "thrust_0_lbs", i),
                "jsb3_thrust_1_lbs": get_array_value(res_3dof, "thrust_1_lbs", i),
                "jsb3_elevator_pos_rad": get_array_value(res_3dof, "elevator_pos_rad", i),
                "jsb3_flap_pos": get_array_value(res_3dof, "flap_pos", i),
                "jsb3_gear_pos": get_array_value(res_3dof, "gear_pos", i),

                "jsb6_throttle_pos_0": get_array_value(res_6dof, "throttle_pos_0", i),
                "jsb6_throttle_pos_1": get_array_value(res_6dof, "throttle_pos_1", i),
                "jsb6_thrust_0_lbs": get_array_value(res_6dof, "thrust_0_lbs", i),
                "jsb6_thrust_1_lbs": get_array_value(res_6dof, "thrust_1_lbs", i),
                "jsb6_elevator_pos_rad": get_array_value(res_6dof, "elevator_pos_rad", i),
                "jsb6_flap_pos": get_array_value(res_6dof, "flap_pos", i),
                "jsb6_gear_pos": get_array_value(res_6dof, "gear_pos", i),

                "own_thrust_N": get_array_value(res_own, "thrust", i),
                "jsb6_total_thrust_N": (
                                               get_array_value(res_6dof, "thrust_0_lbs", i)
                                               + get_array_value(res_6dof, "thrust_1_lbs", i)
                                       ) * LBS_TO_N,

                "own_CL": get_array_value(res_own, "CL", i),
                "own_CD": get_array_value(res_own, "CD", i),
                "own_load_factor": get_array_value(res_own, "load_factor", i),
                "own_vertical_speed": get_array_value(res_own, "vertical_speed", i),

                "own_Cm": get_array_value(res_own, "Cm", i),
                "own_delta_e_rad": get_array_value(res_own, "delta_e_rad", i),
                "own_q_dyn": get_array_value(res_own, "q_dyn", i),
                "own_dq_est": get_array_value(res_own, "dq_est", i),
                "jsb6_dq_est": get_array_value(res_6dof, "dq_est", i),

                "own_load_factor_est": get_array_value(res_own, "load_factor_est", i),
                "jsb6_load_factor_est": get_array_value(res_6dof, "load_factor_est", i),
                "own_dgamma_est": get_array_value(res_own, "dgamma_est", i),
                "jsb6_dgamma_est": get_array_value(res_6dof, "dgamma_est", i),
            }

            writer.writerow(row)


def summarize_scenario_comparison(scenario_name, res_own, res_6dof):
    n = min(len(res_own["t"]), len(res_6dof["t"]))

    if n == 0:
        print(f"No data for scenario {scenario_name}")
        return

    def max_abs_err(key):
        return float(np.max(np.abs(res_own[key][:n] - res_6dof[key][:n])))

    print(f"\n=== Scenario comparison: {scenario_name} ===")
    print(f"max |V_own - V_jsb6|         = {max_abs_err('V'):.4f} m/s")
    print(f"max |alpha_own - alpha_jsb6| = {max_abs_err('alpha'):.4f} deg")
    print(f"max |h_own - h_jsb6|         = {max_abs_err('h'):.4f} m")
    print(f"max |theta_own - theta_jsb6| = {max_abs_err('theta'):.4f} deg")
    print(f"max |gamma_own - gamma_jsb6| = {max_abs_err('gamma'):.4f} deg")
    print(f"max |q_own - q_jsb6|         = {max_abs_err('q'):.6f} rad/s")

    print("\nFinal state difference:")
    print(f"V error       = {res_own['V'][n-1] - res_6dof['V'][n-1]:+.4f} m/s")
    print(f"alpha error   = {res_own['alpha'][n-1] - res_6dof['alpha'][n-1]:+.4f} deg")
    print(f"h error       = {res_own['h'][n-1] - res_6dof['h'][n-1]:+.4f} m")
    print(f"theta error   = {res_own['theta'][n-1] - res_6dof['theta'][n-1]:+.4f} deg")
    print(f"gamma error   = {res_own['gamma'][n-1] - res_6dof['gamma'][n-1]:+.4f} deg")
    print(f"q error       = {res_own['q'][n-1] - res_6dof['q'][n-1]:+.6f} rad/s")


def plot_scenario_comparison(scenario, res_own, res_3dof, res_6dof):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    fig.suptitle(
        f"Сравнение моделей: {scenario.name}\n{scenario.description}",
        fontsize=12
    )

    axs[0, 0].plot(res_3dof["t"], res_3dof["V"], label="JSBSim 3DOF")
    axs[0, 0].plot(res_6dof["t"], res_6dof["V"], label="JSBSim 6DOF")
    axs[0, 0].plot(res_own["t"], res_own["V"], label="Наша модель")
    axs[0, 0].set_ylabel("V, м/с")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(res_3dof["t"], res_3dof["alpha"], label="JSBSim 3DOF")
    axs[0, 1].plot(res_6dof["t"], res_6dof["alpha"], label="JSBSim 6DOF")
    axs[0, 1].plot(res_own["t"], res_own["alpha"], label="Наша модель")
    axs[0, 1].set_ylabel("α, град")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(res_3dof["t"], res_3dof["h"], label="JSBSim 3DOF")
    axs[1, 0].plot(res_6dof["t"], res_6dof["h"], label="JSBSim 6DOF")
    axs[1, 0].plot(res_own["t"], res_own["h"], label="Наша модель")
    axs[1, 0].set_ylabel("h, м")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(res_3dof["t"], res_3dof["theta"], label="JSBSim 3DOF")
    axs[1, 1].plot(res_6dof["t"], res_6dof["theta"], label="JSBSim 6DOF")
    axs[1, 1].plot(res_own["t"], res_own["theta"], label="Наша модель")
    axs[1, 1].set_ylabel("θ, град")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    axs[2, 0].plot(res_3dof["t"], res_3dof["gamma"], label="JSBSim 3DOF")
    axs[2, 0].plot(res_6dof["t"], res_6dof["gamma"], label="JSBSim 6DOF")
    axs[2, 0].plot(res_own["t"], res_own["gamma"], label="Наша модель")
    axs[2, 0].set_ylabel("γ, град")
    axs[2, 0].set_xlabel("t, с")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(res_3dof["t"], res_3dof["q"], label="JSBSim 3DOF")
    axs[2, 1].plot(res_6dof["t"], res_6dof["q"], label="JSBSim 6DOF")
    axs[2, 1].plot(res_own["t"], res_own["q"], label="Наша модель")
    axs[2, 1].set_ylabel("q, рад/с")
    axs[2, 1].set_xlabel("t, с")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    plt.tight_layout()

    out_dir = os.path.join("results", "jsbsim_validation", "scenario_plots")
    os.makedirs(out_dir, exist_ok=True)

    fig_path = os.path.join(out_dir, f"{scenario.name}.png")
    plt.savefig(fig_path, dpi=200)

    print(f"Saved plot to: {fig_path}")

    plt.show()

def print_jsbsim_actuator_diagnostics(name, res):
    print(f"\n=== JSBSim actuator/propulsion diagnostics: {name} ===")

    for key in [
        "throttle",
        "throttle_pos_0",
        "throttle_pos_1",
        "thrust_0_lbs",
        "thrust_1_lbs",
        "elevator_pos_rad",
        "flap_pos",
        "gear_pos",
    ]:
        if key not in res:
            print(f"{key:24s}: нет")
            continue

        arr = np.asarray(res[key], dtype=float)
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            print(f"{key:24s}: all NaN")
            continue

        print(
            f"{key:24s}: "
            f"start={finite[0]: .6f}, "
            f"mean={np.mean(finite): .6f}, "
            f"min={np.min(finite): .6f}, "
            f"max={np.max(finite): .6f}"
        )


def run_validation_scenario(scenario_name: str, show_plot=True):
    scenario = get_scenario_by_name(scenario_name)


    print(f"\n=== Running validation scenario: {scenario.name} ===")
    print(scenario.description)

    print("Running JSBSim 3DOF...")
    res_3dof = run_jsbsim_scenario(scenario, rotation_mode="3dof")
    print_jsbsim_actuator_diagnostics("JSBSim 3DOF", res_3dof)

    print("Running JSBSim 6DOF...")
    res_6dof = run_jsbsim_scenario(scenario, rotation_mode="6dof")
    print("\n=== JSBSim actual initial state ===")
    print(f"V0 jsb      = {res_6dof['V'][0]:.3f}")
    print(f"h0 jsb      = {res_6dof['h'][0]:.3f}")
    print(f"theta0 jsb  = {res_6dof['theta'][0]:.3f}")
    print(f"gamma0 jsb  = {res_6dof['gamma'][0]:.3f}")
    print(f"alpha0 jsb  = {res_6dof['alpha'][0]:.3f}")
    print(f"u0 jsb      = {res_6dof['u'][0]:.6f}")
    print(f"thr0 jsb    = {res_6dof['throttle'][0]:.6f}")
    print_jsbsim_actuator_diagnostics("JSBSim 6DOF", res_6dof)

    add_dq_estimate_to_result(res_6dof)
    add_gamma_derivative_and_load_factor_estimate(res_6dof)

    print("Running own reduced-order model with replayed JSBSim controls...")
    print("\n=== Scenario intended initial state ===")
    s0 = scenario.make_initial_state()
    print(f"V0 own scenario      = {s0.V:.3f}")
    print(f"h0 own scenario      = {s0.h:.3f}")
    print(f"theta0 own scenario  = {np.degrees(s0.theta):.3f}")
    print(f"gamma0 own scenario  = {np.degrees(s0.gamma):.3f}")
    print(f"alpha0 own scenario  = {np.degrees(s0.theta - s0.gamma):.3f}")

    u0, thr0 = scenario.control_law(0.0, s0)
    print(f"u0 scenario          = {u0:.6f}")
    print(f"throttle0 scenario   = {thr0:.6f}")

    res_own = run_own_model_with_replayed_controls(scenario, res_6dof)

    add_gamma_derivative_and_load_factor_estimate(res_own)

    event_rows = [
        compute_event_metrics(res_own, scenario, "own"),
        compute_event_metrics(res_6dof, scenario, "jsbsim_6dof"),
    ]

    print_event_comparison(event_rows)

    event_dir = os.path.join("results", "jsbsim_validation", "event_metrics")
    os.makedirs(event_dir, exist_ok=True)

    event_csv_path = os.path.join(event_dir, f"{scenario.name}_events.csv")
    save_event_metrics_csv(event_csv_path, event_rows)

    print(f"Saved event metrics to: {event_csv_path}")

    summarize_scenario_comparison(
        scenario_name=scenario.name,
        res_own=res_own,
        res_6dof=res_6dof
    )

    out_dir = os.path.join("results", "jsbsim_validation", "scenario_runs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{scenario.name}.csv")

    save_scenario_comparison_csv(
        path=csv_path,
        scenario_name=scenario.name,
        res_own=res_own,
        res_3dof=res_3dof,
        res_6dof=res_6dof
    )

    print(f"Saved CSV to: {csv_path}")

    if show_plot:
        plot_scenario_comparison(
            scenario=scenario,
            res_own=res_own,
            res_3dof=res_3dof,
            res_6dof=res_6dof
        )

    return res_own, res_3dof, res_6dof, event_rows


def run_all_validation_scenarios(show_plot=False):
    for scenario in SCENARIOS:
        run_validation_scenario(
            scenario_name=scenario.name,
            show_plot=show_plot
        )

def get_stall_entry_scenarios(primary_only=False):
    """
    Возвращает сценарии третьей группы: stall_entry.

    primary_only=False:
        берём все сценарии группы 3, включая stress-test / accident-inspired.

    primary_only=True:
        берём только сценарии, помеченные primary_for_validation=True.
    """

    scenarios = [
        scenario for scenario in SCENARIOS
        if getattr(scenario, "group", None) == "stall_entry"
    ]

    if primary_only:
        scenarios = [
            scenario for scenario in scenarios
            if getattr(scenario, "primary_for_validation", False)
        ]

    return scenarios


def run_stall_entry_validation_scenarios(
    show_plot=False,
    primary_only=False,
    aggregate_filename="stall_entry_event_metrics_all.csv",
):
    """
    Запускает все сценарии третьей группы и сохраняет единый CSV
    с событийными метриками own vs JSBSim 6DOF.

    Для каждого сценария также сохраняются:
    - отдельный event CSV;
    - trajectory CSV;
    - plot, если show_plot=True.
    """

    scenarios = get_stall_entry_scenarios(primary_only=primary_only)

    if not scenarios:
        print("No stall_entry scenarios found.")
        return []

    print("\n=== Running stall-entry validation scenarios ===")
    print(f"primary_only = {primary_only}")
    print("Scenarios:")
    for scenario in scenarios:
        print(f"  - {scenario.name}")

    all_event_rows = []

    for scenario in scenarios:
        print("\n" + "=" * 80)
        print(f"RUNNING GROUP-3 SCENARIO: {scenario.name}")
        print("=" * 80)

        try:
            res_own, res_3dof, res_6dof, event_rows = run_validation_scenario(
                scenario_name=scenario.name,
                show_plot=show_plot,
            )

            error_row = compute_event_error_row(event_rows, scenario)
            event_rows_with_error = event_rows + [error_row]

            for row in event_rows_with_error:
                row["scenario_group"] = getattr(scenario, "group", "")
                row["scenario_purpose"] = getattr(scenario, "purpose", "")
                row["parent_trim"] = getattr(scenario, "parent_trim", "")
                row["primary_for_validation"] = getattr(
                    scenario,
                    "primary_for_validation",
                    False
                )

            all_event_rows.extend(event_rows_with_error)

        except Exception as exc:
            print(f"\nERROR while running scenario {scenario.name}: {exc}")

            all_event_rows.append({
                "scenario": scenario.name,
                "model": "ERROR",
                "scenario_group": getattr(scenario, "group", ""),
                "scenario_purpose": getattr(scenario, "purpose", ""),
                "parent_trim": getattr(scenario, "parent_trim", ""),
                "primary_for_validation": getattr(
                    scenario,
                    "primary_for_validation",
                    False
                ),
                "error": str(exc),
            })

    out_dir = os.path.join("results", "jsbsim_validation", "event_metrics")
    os.makedirs(out_dir, exist_ok=True)

    aggregate_path = os.path.join(out_dir, aggregate_filename)

    save_event_metrics_csv(
        path=aggregate_path,
        rows=all_event_rows,
    )

    print("\n=== Saved aggregate event metrics ===")
    print(aggregate_path)

    return all_event_rows


def fit_cm_delta_from_jsbsim(samples):
    """
    samples: список словарей вида:
    {
        "delta_rad": ...,
        "dq_jsb": ...,
        "q_dyn": ...,
        "S": ...,
        "c": ...,
        "Iyy": ...
    }
    """

    deltas = []
    cm_values = []

    for s in samples:
        Cm_required = (
            s["dq_jsb"]
            * s["Iyy"]
            / (s["q_dyn"] * s["S"] * s["c"])
        )

        deltas.append(s["delta_rad"])
        cm_values.append(Cm_required)

    deltas = np.array(deltas)
    cm_values = np.array(cm_values)

    # Cm = a + b * delta
    b, a = np.polyfit(deltas, cm_values, deg=1)

    print("\n=== JSBSim fitted pitch moment model ===")
    print(f"Cm_intercept  = {a:+.6f}")
    print(f"Cm_delta_e    = {b:+.6f} 1/rad")

    return a, b

def compute_required_Cm_from_dq(dq_jsb, q_dyn, S, c, Iyy):
    """
    Пересчитывает угловое ускорение JSBSim в требуемый коэффициент тангажного момента.

    Используется формула:
        M = q_dyn * S * c * Cm
        dq = M / Iyy

    Отсюда:
        Cm = dq * Iyy / (q_dyn * S * c)
    """

    denom = q_dyn * S * c

    if abs(denom) < 1e-9:
        raise ValueError("q_dyn * S * c is too small, cannot compute Cm.")

    return dq_jsb * Iyy / denom

def fit_pitch_moment_model_from_jsbsim(samples):
    """
    Подгоняет локальную линейную модель тангажного момента по данным JSBSim:

        Cm = Cm0 + Cm_alpha * alpha + Cm_delta_e * delta_e

    samples: список словарей вида:
    {
        "alpha_rad": ...,
        "delta_rad": ...,
        "dq_jsb": ...,
        "q_dyn": ...,
        "S": ...,
        "c": ...,
        "Iyy": ...
    }
    """

    X = []
    y = []

    for s in samples:
        Cm_required = compute_required_Cm_from_dq(
            dq_jsb=s["dq_jsb"],
            q_dyn=s["q_dyn"],
            S=s["S"],
            c=s["c"],
            Iyy=s["Iyy"],
        )

        X.append([
            1.0,
            s["alpha_rad"],
            s["delta_rad"],
        ])

        y.append(Cm_required)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # y = X @ coef
    # coef = [Cm0, Cm_alpha, Cm_delta_e]
    coef, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)

    Cm0, Cm_alpha, Cm_delta_e = coef

    y_pred = X @ coef
    errors = y - y_pred

    mean_abs_error = float(np.mean(np.abs(errors)))
    max_abs_error = float(np.max(np.abs(errors)))

    print("\n=== JSBSim fitted pitch moment model ===")
    print(f"Cm0        = {Cm0:+.6f}")
    print(f"Cm_alpha   = {Cm_alpha:+.6f} 1/rad")
    print(f"Cm_delta_e = {Cm_delta_e:+.6f} 1/rad")
    print()
    print(f"mean abs Cm error = {mean_abs_error:.6f}")
    print(f"max abs Cm error  = {max_abs_error:.6f}")
    print(f"rank              = {rank}")

    print("\n=== Recommended own-model update ===")
    print(f"BOEING_737_JSBSIM.Cm0        = {Cm0:+.6f}")
    print(f"BOEING_737_JSBSIM.Cm_alpha   = {Cm_alpha:+.6f}")
    print(f"BOEING_737_JSBSIM.Cm_delta_e = {Cm_delta_e:+.6f}")

    return PitchMomentFitResult(
        Cm0=float(Cm0),
        Cm_alpha=float(Cm_alpha),
        Cm_delta_e=float(Cm_delta_e),
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
    )


def collect_pitch_moment_samples_from_jsbsim(
        alpha_grid_deg=None,
        elevator_grid_deg=None,
        V0=80.0,
        h0=500.0,
        gamma0_deg=-3.92,
        rotation_mode="6dof",
        elevator_max_rad=0.30,
):
    """
    Собирает набор точек JSBSim для подгонки локальной модели:

        Cm = Cm0 + Cm_alpha * alpha + Cm_delta_e * delta_e

    Важно:
    - alpha задаётся через ic/alpha-deg;
    - theta задаётся как gamma + alpha;
    - elevator_grid_deg задаёт именно желаемое фактическое отклонение руля;
    - в sample сохраняется фактический elevator-pos-rad из JSBSim, а не только команда.
    """

    if alpha_grid_deg is None:
        alpha_grid_deg = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0]

    if elevator_grid_deg is None:
        elevator_grid_deg = [-10.0, -5.0, 0.0, 5.0, 10.0]

    samples = []

    aircraft = BOEING_737_JSBSIM
    dynamics = AircraftDynamics(aircraft, CLEAN)

    print("\n=== Collecting JSBSim pitch moment samples ===")
    print(
        f"{'alpha_cmd':>10} | "
        f"{'alpha_act':>10} | "
        f"{'elev_cmd':>10} | "
        f"{'elev_act':>10} | "
        f"{'dq_jsb':>12} | "
        f"{'Cm_req':>12}"
    )
    print("-" * 78)

    for alpha_deg in alpha_grid_deg:
        for elevator_deg in elevator_grid_deg:
            gamma0_rad = np.radians(gamma0_deg)
            alpha_cmd_rad = np.radians(alpha_deg)
            theta0_rad = gamma0_rad + alpha_cmd_rad
            theta0_deg = np.degrees(theta0_rad)

            desired_delta_rad = np.radians(elevator_deg)

            # В JSBSim для твоей модели было видно:
            # elevator-pos-rad ≈ elevator-cmd-norm * 0.30
            # Поэтому переводим желаемый delta_rad в нормированную команду.
            u_cmd = desired_delta_rad / elevator_max_rad
            u_cmd = float(np.clip(u_cmd, -1.0, 1.0))

            fdm = FGFDMExec(JSBSIM_DATA_PATH)
            fdm.load_model("737")
            fdm.set_dt(dt)

            # Начальные условия
            fdm.set_property_value("ic/h-sl-ft", h0 / 0.3048)
            fdm.set_property_value("ic/vt-fps", V0 / 0.3048)
            fdm.set_property_value("ic/alpha-deg", alpha_deg)
            fdm.set_property_value("ic/theta-deg", theta0_deg)
            fdm.set_property_value("ic/gamma-deg", gamma0_deg)

            fdm.set_property_value("ic/phi-deg", 0.0)
            fdm.set_property_value("ic/psi-deg", 0.0)
            fdm.set_property_value("ic/lat-gc-deg", 0.0)
            fdm.set_property_value("ic/long-gc-deg", 0.0)

            if rotation_mode == "3dof":
                fdm.set_property_value("simulation/rotation", 0)
            else:
                fdm.set_property_value("simulation/rotation", 1)

            fdm.run_ic()

            # Задаём управление.
            fdm.set_property_value("fcs/elevator-cmd-norm", u_cmd)
            fdm.set_property_value("fcs/throttle-cmd-norm", throttle)

            # Первый шаг нужен, чтобы JSBSim/FCS применил команду.
            fdm.run()

            before = read_jsbsim_snapshot(fdm)

            # Читаем именно фактическое положение руля.
            delta_rad = try_get(fdm, "fcs/elevator-pos-rad")

            if delta_rad is None:
                delta_rad = try_get(fdm, "fcs/elevator-control")

            if delta_rad is None:
                raise RuntimeError(
                    "Не удалось прочитать фактическое положение elevator в JSBSim."
                )

            # Второй шаг — для оценки производной dq.
            fdm.run()
            after = read_jsbsim_snapshot(fdm)

            der = finite_difference(before, after)

            # Используем ту же q_dyn-схему, что и в собственной модели.
            rho = dynamics.atmosphere.density(before["h"])
            q_dyn = 0.5 * rho * before["V"] ** 2

            sample = {
                "alpha_cmd_deg": alpha_deg,
                "alpha_deg": np.degrees(before["alpha"]),
                "alpha_rad": before["alpha"],

                "elevator_cmd_deg": elevator_deg,
                "u_cmd": u_cmd,
                "delta_rad": delta_rad,
                "delta_deg": np.degrees(delta_rad),

                "dq_jsb": der["dq"],
                "q_dyn": q_dyn,
                "S": aircraft.wing_area,
                "c": aircraft.mean_chord,
                "Iyy": aircraft.Iyy,

                "V": before["V"],
                "theta_deg": np.degrees(before["theta"]),
                "gamma_deg": np.degrees(before["gamma"]),
                "h": before["h"],
            }

            Cm_required = compute_required_Cm_from_dq(
                dq_jsb=sample["dq_jsb"],
                q_dyn=sample["q_dyn"],
                S=sample["S"],
                c=sample["c"],
                Iyy=sample["Iyy"],
            )

            sample["Cm_required"] = Cm_required
            samples.append(sample)

            print(
                f"{alpha_deg:>10.3f} | "
                f"{sample['alpha_deg']:>10.3f} | "
                f"{elevator_deg:>10.3f} | "
                f"{sample['delta_deg']:>10.3f} | "
                f"{sample['dq_jsb']:>12.6f} | "
                f"{Cm_required:>12.6f}"
            )

    return samples

def read_jsbsim_snapshot(fdm):
    """
    Считывает минимальный набор переменных из JSBSim
    и приводит их к СИ.
    """
    V = fdm.get_property_value("velocities/vt-fps") * 0.3048
    alpha = np.radians(fdm.get_property_value("aero/alpha-deg"))
    theta = np.radians(fdm.get_property_value("attitude/theta-deg"))
    h = fdm.get_property_value("position/h-sl-ft") * 0.3048

    # Для продольной модели можно восстановить gamma как theta - alpha
    # при малых углах скольжения и продольном движении.
    gamma = theta - alpha

    try:
        q = fdm.get_property_value("velocities/q-rad_sec")
    except Exception:
        q = np.nan

    try:
        sim_time = fdm.get_property_value("simulation/sim-time-sec")
    except Exception:
        sim_time = np.nan

    return {
        "t": sim_time,
        "V": V,
        "alpha": alpha,
        "theta": theta,
        "gamma": gamma,
        "h": h,
        "q": q,
    }

def finite_difference(before, after):
    """
    Считает производные по двум последовательным состояниям.
    """
    dt_actual = after["t"] - before["t"]

    if not np.isfinite(dt_actual) or dt_actual <= 0:
        dt_actual = dt

    dV = (after["V"] - before["V"]) / dt_actual
    dtheta = (after["theta"] - before["theta"]) / dt_actual
    dgamma = (after["gamma"] - before["gamma"]) / dt_actual
    dh = (after["h"] - before["h"]) / dt_actual

    if np.isfinite(before["q"]) and np.isfinite(after["q"]):
        dq = (after["q"] - before["q"]) / dt_actual
    else:
        dq = np.nan

    return {
        "dt": dt_actual,
        "dV": dV,
        "dtheta": dtheta,
        "dgamma": dgamma,
        "dh": dh,
        "dq": dq,
    }

def try_get(fdm, name):
    try:
        return fdm.get_property_value(name)
    except Exception:
        return None


def print_elevator_properties(fdm):
    candidates = [
        "fcs/elevator-cmd-norm",
        "fcs/elevator-pos-norm",
        "fcs/elevator-pos-rad",
        "fcs/elevator-pos-deg",
        "fcs/elevator-control",
        "fcs/elevator-control-norm",
        "fcs/elevator-trim-cmd-norm",
        "fcs/pitch-trim-cmd-norm",
    ]

    print("\n=== JSBSim elevator properties ===")
    for name in candidates:
        value = try_get(fdm, name)
        if value is not None:
            print(f"{name:30s} = {value}")





def collect_jsbsim_pitch_sample(u_cmd, rotation_mode="6dof"):
    """
    Запускает JSBSim из одной и той же начальной точки,
    задаёт elevator command и возвращает одну точку для fit:
    delta_rad -> dq_jsb -> Cm_required.
    """

    fdm = FGFDMExec(JSBSIM_DATA_PATH)
    fdm.load_model("737")
    fdm.set_dt(dt)

    # Начальные условия
    fdm.set_property_value("ic/h-sl-ft", h0 / 0.3048)
    fdm.set_property_value("ic/vt-fps", V0 / 0.3048)
    fdm.set_property_value("ic/alpha-deg", alpha0)
    fdm.set_property_value("ic/theta-deg", theta0)
    fdm.set_property_value("ic/gamma-deg", gamma0)

    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-deg", 0.0)
    fdm.set_property_value("ic/lat-gc-deg", 0.0)
    fdm.set_property_value("ic/long-gc-deg", 0.0)

    if rotation_mode == "3dof":
        fdm.set_property_value("simulation/rotation", 0)
    else:
        fdm.set_property_value("simulation/rotation", 1)

    fdm.run_ic()

    # Задаём управление
    fdm.set_property_value("fcs/elevator-cmd-norm", u_cmd)
    fdm.set_property_value("fcs/throttle-cmd-norm", throttle)

    # Один шаг нужен, чтобы FCS применил команду руля.
    # Иначе можно случайно измерить не то положение elevator.
    fdm.run()

    before = read_jsbsim_snapshot(fdm)

    # Фактическое положение руля, а не только команда
    delta_rad = try_get(fdm, "fcs/elevator-pos-rad")
    if delta_rad is None:
        delta_rad = try_get(fdm, "fcs/elevator-control")

    if delta_rad is None:
        raise RuntimeError("Не удалось прочитать фактическое положение elevator в JSBSim.")

    # Второй шаг — для оценки производных
    fdm.run()
    after = read_jsbsim_snapshot(fdm)

    der = finite_difference(before, after)

    # Берём q_dyn в той же системе масштабирования, что и в собственной модели
    aircraft = BOEING_737_JSBSIM
    dynamics = AircraftDynamics(aircraft, CLEAN)

    rho = dynamics.atmosphere.density(before["h"])
    q_dyn = 0.5 * rho * before["V"] ** 2

    sample = {
        "u_cmd": u_cmd,
        "delta_rad": delta_rad,
        "delta_deg": np.degrees(delta_rad),
        "dq_jsb": der["dq"],
        "q_dyn": q_dyn,
        "S": aircraft.wing_area,
        "c": aircraft.mean_chord,
        "Iyy": aircraft.Iyy,
        "V": before["V"],
        "alpha_deg": np.degrees(before["alpha"]),
        "theta_deg": np.degrees(before["theta"]),
        "gamma_deg": np.degrees(before["gamma"]),
    }

    return sample
def run_elevator_cm_sweep(rotation_mode="6dof"):
    """
    Прогоняет JSBSim при разных отклонениях руля высоты
    и восстанавливает Cm_delta_e.
    """

    u_values = [-0.5, -0.35, -0.2, 0.0, 0.2, 0.35, 0.5]

    samples = []

    print()
    print("=== JSBSim elevator Cm sweep ===")
    print(
        f"{'u_cmd':>8} | "
        f"{'delta_deg':>10} | "
        f"{'dq_jsb':>12} | "
        f"{'Cm_required':>14}"
    )
    print("-" * 56)

    for u_cmd in u_values:
        sample = collect_jsbsim_pitch_sample(
            u_cmd=u_cmd,
            rotation_mode=rotation_mode
        )

        Cm_required = (
            sample["dq_jsb"]
            * sample["Iyy"]
            / max(sample["q_dyn"] * sample["S"] * sample["c"], 1e-9)
        )

        sample["Cm_required"] = Cm_required
        samples.append(sample)

        print(
            f"{sample['u_cmd']:>8.3f} | "
            f"{sample['delta_deg']:>10.3f} | "
            f"{sample['dq_jsb']:>12.6f} | "
            f"{Cm_required:>14.6f}"
        )

    Cm_intercept, Cm_delta_e = fit_cm_delta_from_jsbsim(samples)

    print()
    print("=== Recommended own-model update ===")
    print(f"Set BOEING_737_JSBSIM.Cm_delta_e ≈ {Cm_delta_e:+.6f}")

    return samples, Cm_intercept, Cm_delta_e


def estimate_required_u_for_cm(dynamics, state, throttle, target_Cm, dt):
    test_values = []

    for u in np.linspace(-1.0, 1.0, 41):
        test_state = deepcopy(state)

        result = dynamics.derivatives(
            test_state,
            u=u,
            throttle=throttle,
            dt=dt
        )

        dtheta, dq, dV, dgamma, dh, dx, dsep, diagnostics = result
        aero, forces, stall, moment, energy, throttle_used = diagnostics

        test_values.append((u, moment.Cm, dq, np.degrees(moment.delta)))

    best = min(test_values, key=lambda item: abs(item[1] - target_Cm))

    print("\n=== Required own-model elevator command ===")
    print(f"target Cm = {target_Cm:+.6f}")
    print(f"best u    = {best[0]:+.4f}")
    print(f"own Cm    = {best[1]:+.6f}")
    print(f"own dq    = {best[2]:+.6f}")
    print(f"delta deg = {best[3]:+.4f}")

    return best

def compare_required_drag(
        dV_jsb,
        gamma,
        mass,
        q_dyn,
        S,
        T_x=0.0,
        g=9.81
):
    """
    Из уравнения:
        dV = (T_x - D) / m - g sin(gamma)

    выражаем сопротивление, которое требуется для совпадения с JSBSim:
        D = T_x - m * (dV + g sin(gamma))
    """

    D_required = T_x - mass * (dV_jsb + g * np.sin(gamma))
    CD_required = D_required / max(q_dyn * S, 1e-9)

    print("\n=== Required drag from JSBSim dV ===")
    print(f"mass               = {mass:.3f} kg")
    print(f"gamma              = {np.degrees(gamma):+.4f} deg")
    print(f"dV_jsb             = {dV_jsb:+.6f} m/s^2")
    print(f"T_x                = {T_x:+.3f} N")
    print(f"D_required         = {D_required:.3f} N")
    print(f"CD_required        = {CD_required:.6f}")

    return D_required, CD_required


def set_if_exists(jsb, prop_name, value):
    """
    Безопасная установка свойства JSBSim.
    Если свойства нет — просто пропускаем.
    """
    try:
        jsb.set_property_value(prop_name, value)
    except Exception:
        pass


def make_own_initial_state_from_jsbsim(jsb):
    """
    Создаёт начальное состояние собственной модели из текущего состояния JSBSim.
    Так мы гарантируем, что обе модели стартуют из одной точки.
    """

    jsb_state = get_jsbsim_state(jsb)

    state = State()
    state.V = jsb_state["V"]
    state.theta = jsb_state["theta"]
    state.gamma = jsb_state["gamma"]
    state.alpha = jsb_state["alpha"]
    state.alpha_prev = jsb_state["alpha"]
    state.q = jsb_state["q"]
    state.h = jsb_state["h"]
    state.x = 0.0
    state.sep = 0.0
    state.R = 0.0

    return state


def apply_jsbsim_controls(jsb, elevator_cmd_norm=-0.315, throttle_cmd=0.0):
    elevator_cmd_norm = float(np.clip(elevator_cmd_norm, -1.0, 1.0))
    throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    set_if_exists(jsb, "fcs/elevator-cmd-norm", elevator_cmd_norm)
    set_if_exists(jsb, "fcs/elevator-pos-norm", elevator_cmd_norm)

    set_if_exists(jsb, "fcs/throttle-cmd-norm", throttle_cmd)
    set_if_exists(jsb, "fcs/throttle-pos-norm", throttle_cmd)

    set_if_exists(jsb, "fcs/throttle-cmd-norm[0]", throttle_cmd)
    set_if_exists(jsb, "fcs/throttle-cmd-norm[1]", throttle_cmd)

    set_if_exists(jsb, "fcs/throttle-pos-norm[0]", throttle_cmd)
    set_if_exists(jsb, "fcs/throttle-pos-norm[1]", throttle_cmd)

    set_if_exists(jsb, "propulsion/engine[0]/throttle", throttle_cmd)
    set_if_exists(jsb, "propulsion/engine[1]/throttle", throttle_cmd)


def read_engine_diagnostics(jsb):
    return {
        "throttle_pos_0": safe_get(jsb, "fcs/throttle-pos-norm[0]", np.nan),
        "throttle_pos_1": safe_get(jsb, "fcs/throttle-pos-norm[1]", np.nan),

        "thrust_0_lbs": safe_get(jsb, "propulsion/engine[0]/thrust-lbs", np.nan),
        "thrust_1_lbs": safe_get(jsb, "propulsion/engine[1]/thrust-lbs", np.nan),

        "thrust_total_lbs": safe_get(jsb, "propulsion/total-thrust-lbs", np.nan),
        "thrust_lbs": safe_get(jsb, "propulsion/thrust-lbs", np.nan),

        "n1_0": safe_get(jsb, "propulsion/engine[0]/n1", np.nan),
        "n1_1": safe_get(jsb, "propulsion/engine[1]/n1", np.nan),

        "n2_0": safe_get(jsb, "propulsion/engine[0]/n2", np.nan),
        "n2_1": safe_get(jsb, "propulsion/engine[1]/n2", np.nan),

        "running_0": safe_get(jsb, "propulsion/engine[0]/running", np.nan),
        "running_1": safe_get(jsb, "propulsion/engine[1]/running", np.nan),

        "cutoff_0": safe_get(jsb, "propulsion/engine[0]/cutoff", np.nan),
        "cutoff_1": safe_get(jsb, "propulsion/engine[1]/cutoff", np.nan),

        "fuel_flow_0": safe_get(jsb, "propulsion/engine[0]/fuel-flow-rate-pps", np.nan),
        "fuel_flow_1": safe_get(jsb, "propulsion/engine[1]/fuel-flow-rate-pps", np.nan),
    }


def print_engine_snapshot(jsb, label):
    d = read_engine_diagnostics(jsb)

    print(f"\n=== Engine snapshot: {label} ===")
    for key, value in d.items():
        print(f"{key:24s} = {value}")


def run_jsbsim(rotation_mode: str):
    fdm = FGFDMExec(JSBSIM_DATA_PATH)
    fdm.load_model("737")
    fdm.set_dt(dt)

    fdm.set_property_value("ic/h-sl-ft", h0 / 0.3048)
    fdm.set_property_value("ic/vt-fps", V0 / 0.3048)
    fdm.set_property_value("ic/alpha-deg", alpha0)
    fdm.set_property_value("ic/theta-deg", theta0)
    fdm.set_property_value("ic/gamma-deg", gamma0)

    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-deg", 0.0)
    fdm.set_property_value("ic/lat-gc-deg", 0.0)
    fdm.set_property_value("ic/long-gc-deg", 0.0)

    if rotation_mode == "3dof":
        fdm.set_property_value("simulation/rotation", 0)
    else:
        fdm.set_property_value("simulation/rotation", 1)

    fdm.run_ic()

    t_arr, V_arr, alpha_arr, h_arr, theta_arr = [], [], [], [], []

    while True:
        t = fdm.get_property_value("simulation/sim-time-sec")

        V = fdm.get_property_value("velocities/vt-fps") * 0.3048
        alpha = fdm.get_property_value("aero/alpha-deg")
        h = fdm.get_property_value("position/h-sl-ft") * 0.3048
        theta = fdm.get_property_value("attitude/theta-deg")

        t_arr.append(t)
        V_arr.append(V)
        alpha_arr.append(alpha)
        h_arr.append(h)
        theta_arr.append(theta)

        fdm.set_property_value("fcs/elevator-cmd-norm", -0.2)
        fdm.set_property_value("fcs/throttle-cmd-norm", throttle)

        fdm.run()

        if t >= t_final or h <= 0.0:
            break

    return {
        "t": np.array(t_arr),
        "V": np.array(V_arr),
        "alpha": np.array(alpha_arr),
        "h": np.array(h_arr),
        "theta": np.array(theta_arr),
    }

def replay_controls_from_reference(t, ref_t, ref_u, ref_throttle):
    u = np.interp(t, ref_t, ref_u)
    throttle = np.interp(t, ref_t, ref_throttle)
    return float(u), float(throttle)


def run_own_model_with_replayed_controls(scenario, ref_result):
    aircraft = BOEING_737_JSBSIM
    config = scenario.config

    dynamics = AircraftDynamics(aircraft, config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    if "initial_state_for_own" in ref_result:
        state = deepcopy(ref_result["initial_state_for_own"])
    else:
        state = scenario.make_initial_state()

    t_arr = []
    V_arr = []
    alpha_arr = []
    h_arr = []
    theta_arr = []
    gamma_arr = []
    q_arr = []
    u_arr = []
    throttle_arr = []
    R_arr = []
    sep_arr = []
    mode_arr = []
    thrust_arr = []
    CL_arr = []
    CD_arr = []
    load_factor_arr = []
    vertical_speed_arr = []
    Cm_arr = []
    delta_e_arr = []
    q_dyn_arr = []
    dq_est_arr = []

    prev_q = None

    t = 0.0

    while t <= scenario.t_final:
        u = replay_actual_elevator_from_jsbsim(
            t,
            ref_result,
            aircraft
        )

        target_thrust_N = replay_jsbsim_total_thrust_N(t, ref_result)

        thr = throttle_for_target_thrust(
            dynamics,
            state,
            target_thrust_N
        )

        state = integrator.step(
            state,
            dynamics,
            risk_model,
            u,
            thr,
            dt
        )
        thrust_arr.append(state.thrust)
        Cm_arr.append(state.Cm)
        delta_e_arr.append(state.elevator_delta)

        rho = dynamics.atmosphere.density(state.h)
        q_dyn = 0.5 * rho * state.V ** 2
        q_dyn_arr.append(q_dyn)

        if prev_q is None:
            dq_est_arr.append(np.nan)
        else:
            dq_est_arr.append((state.q - prev_q) / dt)

        prev_q = state.q

        CL_arr.append(state.CL)
        CD_arr.append(state.CD)
        load_factor_arr.append(state.load_factor)
        vertical_speed_arr.append(state.vertical_speed)

        alpha = dynamics.compute_alpha(state.theta, state.gamma)

        t_arr.append(t)
        V_arr.append(state.V)
        alpha_arr.append(np.degrees(alpha))
        h_arr.append(state.h)
        theta_arr.append(np.degrees(state.theta))
        gamma_arr.append(np.degrees(state.gamma))
        q_arr.append(state.q)
        u_arr.append(u)
        throttle_arr.append(thr)
        R_arr.append(state.R)
        sep_arr.append(state.sep)
        mode_arr.append(state.mode)

        t += dt

        if state.h <= 0.0 or state.V <= 5.0:
            break

    return {
        "t": np.array(t_arr),
        "V": np.array(V_arr),
        "alpha": np.array(alpha_arr),
        "h": np.array(h_arr),
        "theta": np.array(theta_arr),
        "gamma": np.array(gamma_arr),
        "q": np.array(q_arr),
        "u": np.array(u_arr),
        "throttle": np.array(throttle_arr),
        "R": np.array(R_arr),
        "sep": np.array(sep_arr),
        "mode": np.array(mode_arr, dtype=object),
        "thrust": np.array(thrust_arr),
        "CL": np.array(CL_arr),
        "CD": np.array(CD_arr),
        "load_factor": np.array(load_factor_arr),
        "vertical_speed": np.array(vertical_speed_arr),
        "Cm": np.array(Cm_arr),
        "delta_e_rad": np.array(delta_e_arr),
        "q_dyn": np.array(q_dyn_arr),
        "dq_est": np.array(dq_est_arr),
    }


def run_own_model():
    """Запуск вашей 3DOF модели с теми же начальными условиями."""
    aircraft = BOEING_737_JSBSIM
    config = CLEAN
    dynamics = AircraftDynamics(aircraft, config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    from model.state import State
    state = State()
    state.V = V0
    state.h = h0
    state.theta = np.radians(theta0)
    state.gamma = np.radians(gamma0)
    state.q = 0.0
    state.x = 0.0
    state.R = 0.0
    state.sep = 0.0

    t_arr, V_arr, alpha_arr, h_arr, theta_arr = [], [], [], [], []
    t = 0.0

    while t <= t_final:
        # фиксированное управление
        u = u_elev
        thr = throttle

        state = integrator.step(state, dynamics, risk_model, u, thr, dt)

        V_arr.append(state.V)
        alpha = dynamics.compute_alpha(state.theta, state.gamma)
        alpha_arr.append(np.degrees(alpha))
        h_arr.append(state.h)
        theta_arr.append(np.degrees(state.theta))
        t_arr.append(t)

        t += dt

        if state.h <= 0.0:
            print("Собственная модель достигла земли")
            break

    return {
        "t": np.array(t_arr),
        "V": np.array(V_arr),
        "alpha": np.array(alpha_arr),
        "h": np.array(h_arr),
        "theta": np.array(theta_arr)
    }


def main():
    # # Один сценарий:
    # run_validation_scenario(
    #     scenario_name="low_speed_climb_trim_737",
    #     show_plot=True
    # )


    # Для пакетной проверки всех сценариев:
    run_all_validation_scenarios(show_plot=True)

# def main():
#     run_stall_entry_validation_scenarios(
#         show_plot=False,
#         primary_only=True,
#         aggregate_filename="stall_entry_event_metrics_all.csv",
#     )


if __name__ == "__main__":
    main()

