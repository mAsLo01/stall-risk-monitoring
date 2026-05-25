# validation/calibrate_landing_config.py

"""
Калибровка посадочной конфигурации LANDING по событийным метрикам.

Цель:
    подобрать параметры LANDING так, чтобы reduced-order модель
    в сценарии landing_approach_to_stall_737 была ближе к JSBSim
    по событиям, а не по полному совпадению траектории.

Проверки для каждой комбинации параметров:
    1. trim собственной модели для landing_low_speed_trim_737;
    2. sanity-check первых 5 секунд landing_approach_to_stall_737;
    3. event metrics own vs JSBSim 6DOF;
    4. objective function.

Основная цель калибровки:
    - убрать ложный stall_proxy в LANDING, если JSBSim его не показывает;
    - приблизить t_alpha_10, t_alpha_12, t_warning_proxy;
    - уменьшить ошибку max_alpha и потери высоты до warning.

Запуск:
    python validation/calibrate_landing_config.py

Результат:
    results/jsbsim_validation/landing_calibration/landing_calibration.csv
"""

import os
import sys
import csv
from copy import copy, deepcopy
from dataclasses import replace, is_dataclass

import numpy as np


# ============================================================
# Пути проекта
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# ============================================================
# Импорты собственной модели и сценариев
# ============================================================

from AircraftModel import BOEING_737_JSBSIM, LANDING
from model.state import State

from trim_737 import find_own_model_trim

from scenario_sanity_5s import (
    run_own_scenario_short,
    summarize_rows,
)

# Если jsbsim_validation.py лежит в корне проекта, этот импорт корректен.
# Если ты перенесёшь его в validation/, замени на:
# from jsbsim_validation import ... при запуске из validation
from jsbsim_validation import (
    get_scenario_by_name,
    run_jsbsim_scenario,
    run_own_model_with_replayed_controls,
    add_gamma_derivative_and_load_factor_estimate,
    compute_event_metrics,
)


# ============================================================
# Настройки калибровки
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "jsbsim_validation",
    "landing_calibration",
)

OUT_CSV = os.path.join(RESULTS_DIR, "landing_calibration.csv")


# Основная моментная сетка.
# Начинаем именно с момента, потому что текущая проблема LANDING —
# переоценка stall / слишком большой рост alpha.
ELEVATOR_GAIN_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
DCM0_GRID = [-0.04, -0.03, -0.02, -0.01, 0.00, 0.01]


# Расширенная сетка по CL/CD.
# По умолчанию отключена. Включай только если моментная калибровка
# не дала приемлемого результата.
USE_EXTENDED_CL_CD_GRID = True

DCL0_GRID = [0.60]
DCD0_GRID = [0.010, 0.015, 0.020]


# Целевые параметры landing trim.
LANDING_TRIM_NAME = "landing_low_speed_trim_737"
LANDING_SCENARIO_NAME = "landing_approach_to_stall_737"

LANDING_TRIM_V = 95.0
LANDING_TRIM_H = 800.0
LANDING_TRIM_GAMMA_DEG = 0.0


# Если True, сценарий landing_approach_to_stall_737 будет каждый раз
# пересобираться от найденного trim для текущих параметров LANDING.
#
# Это правильнее для калибровки: если dCm0/elevator_gain изменились,
# старые theta/u/throttle уже могут быть не trim.
USE_DYNAMIC_TRIM_IN_SCENARIO = False


# Сценарный pitch/throttle schedule для LANDING.
# Первая точка будет автоматически заменена на найденный trim:
# theta=trim["theta_deg"], throttle=trim["throttle"].
#
# Остальные точки задают возмущение после первых 5 секунд.
LANDING_SCHEDULE_AFTER_TRIM = [
    # t, theta_cmd_deg, throttle_cmd
    # первые 5 секунд держим trim, поэтому первая точка возмущения позже 5 секунд
    (12.0, 5.0, 0.22),
    (20.0, 8.0, 0.18),
    (28.0, 12.0, 0.15),
]


# Параметры простого theta-controller для сценария.
# Если твой исходный scheduled_trim_controller использовал другие Kp/Kd,
# можешь поставить их здесь.
THETA_KP = 8.0
THETA_KD = 4.0


# ============================================================
# Утилиты
# ============================================================

def safe_float(value, default=np.nan):
    if value is None:
        return default

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(value):
        return default

    return value


def safe_abs(value, default=1000.0):
    value = safe_float(value, default=np.nan)
    if not np.isfinite(value):
        return default
    return abs(value)


def event_error_penalty(error_row, key, missing_penalty=100.0):
    own_key = "own_" + key
    jsb_key = "jsb_" + key

    err_key = "err_" + key

    err = error_row.get(err_key)

    if err is not None:
        try:
            if np.isfinite(err):
                return abs(float(err))
        except TypeError:
            pass

    own_has = error_row.get(own_key) is not None
    jsb_has = error_row.get(jsb_key) is not None

    if not own_has and not jsb_has:
        return 0.0

    return missing_penalty


def save_rows_csv(path, rows, delimiter=";"):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def make_landing_config(
    elevator_moment_gain,
    dCm0,
    dCL0=None,
    dCd0=None,
):
    """
    Создаёт временную копию LANDING с изменёнными параметрами.

    Не меняет глобальный LANDING в AircraftModel.py.
    """

    if is_dataclass(LANDING):
        cfg = replace(LANDING)
    else:
        cfg = copy(LANDING)

    cfg.elevator_moment_gain = float(elevator_moment_gain)
    cfg.dCm0 = float(dCm0)

    if dCL0 is not None:
        cfg.dCL0 = float(dCL0)

    if dCd0 is not None:
        cfg.dCd0 = float(dCd0)

    return cfg


def make_state_from_trim(trim):
    """
    Создаёт State из найденного own-trim.
    """

    state = State()

    state.V = float(trim["V"])
    state.h = float(trim["h"])
    state.theta = np.radians(float(trim["theta_deg"]))
    state.gamma = np.radians(float(trim["gamma_deg"]))
    state.alpha = state.theta - state.gamma
    state.alpha_prev = state.alpha
    state.q = 0.0

    state.x = 0.0
    state.sep = 0.0
    state.R = 0.0

    return state


def interpolate_schedule(t, schedule):
    """
    Линейная интерполяция schedule по времени.

    schedule:
        list of tuples (time, theta_cmd_deg, throttle_cmd)
    """

    if t <= schedule[0][0]:
        return schedule[0][1], schedule[0][2]

    for i in range(1, len(schedule)):
        t0, theta0, thr0 = schedule[i - 1]
        t1, theta1, thr1 = schedule[i]

        if t <= t1:
            w = (t - t0) / max(t1 - t0, 1e-9)

            theta = theta0 * (1.0 - w) + theta1 * w
            throttle = thr0 * (1.0 - w) + thr1 * w

            return theta, throttle

    return schedule[-1][1], schedule[-1][2]


def make_landing_control_law_from_trim(trim):
    """
    Создаёт control_law для landing-сценария на основе найденного trim.

    Первые 5 секунд сценарий остаётся в trim:
        theta_cmd = trim theta
        throttle = trim throttle

    Далее применяется заданный LANDING_SCHEDULE_AFTER_TRIM.
    """

    theta_trim_deg = float(trim["theta_deg"])
    u_trim = float(trim["u"])
    throttle_trim = float(trim["throttle"])

    schedule = [
                   (0.0, theta_trim_deg, throttle_trim),
                   (5.0, theta_trim_deg, throttle_trim),
               ] + [
                   point for point in LANDING_SCHEDULE_AFTER_TRIM
                   if point[0] > 5.0
               ]

    def control_law(t, state):
        theta_cmd_deg, throttle_cmd = interpolate_schedule(t, schedule)

        theta_cmd = np.radians(theta_cmd_deg)

        # Простой PD по тангажу.
        # q_cmd = 0, поэтому производная ошибка = -state.q.
        u = (
            u_trim
            + THETA_KP * (theta_cmd - state.theta)
            - THETA_KD * state.q
        )

        u = float(np.clip(u, -1.0, 1.0))
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

        return u, throttle_cmd

    return control_law


def make_calibration_scenario(base_scenario, landing_cfg, trim=None):
    """
    Создаёт копию landing-сценария с новой LANDING-конфигурацией.

    Если USE_DYNAMIC_TRIM_IN_SCENARIO=True, то:
        - make_initial_state берётся из найденного trim;
        - control_law строится от найденного trim.

    Если False:
        - используется исходный make_initial_state/control_law,
          но config заменяется на landing_cfg.
    """

    scenario = deepcopy(base_scenario)
    scenario.config = landing_cfg

    if USE_DYNAMIC_TRIM_IN_SCENARIO:
        if trim is None:
            raise ValueError("trim is required when USE_DYNAMIC_TRIM_IN_SCENARIO=True")

        scenario.make_initial_state = lambda: make_state_from_trim(trim)
        scenario.control_law = make_landing_control_law_from_trim(trim)

    return scenario


def compute_event_error_row_local(event_rows, scenario):
    """
    Локальная версия compute_event_error_row:
        error = own - jsbsim_6dof

    Нужна, чтобы calibrate_landing_config.py работал даже если
    в jsbsim_validation.py эта функция ещё не добавлена.
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


def compute_landing_objective(error_row):
    """
    Целевая функция для LANDING.

    Главный штраф:
        - mismatch фазы;
        - ложный stall в own, если JSBSim stall не показывает.

    Затем:
        - ошибка t_warning_proxy;
        - ошибки t_alpha_10/t_alpha_12;
        - ошибка max_alpha;
        - ошибка потери высоты до warning.
    """

    own_stall = bool(error_row.get("own_stall_reached", False))
    jsb_stall = bool(error_row.get("jsb_stall_reached", False))

    own_phase = error_row.get("own_phase_reached")
    jsb_phase = error_row.get("jsb_phase_reached")

    phase_mismatch = 0.0 if own_phase == jsb_phase else 1.0
    false_stall = 1.0 if own_stall and not jsb_stall else 0.0

    obj = 0.0

    # Большие штрафы за неправильный класс события.
    obj += 100.0 * phase_mismatch
    obj += 300.0 * false_stall

    # Временные события.
    obj += 10.0 * safe_abs(error_row.get("err_t_warning_proxy"))
    obj += 5.0 * safe_abs(error_row.get("err_t_alpha_10"))
    err_a12 = error_row.get("err_t_alpha_12")
    if err_a12 is not None:
        obj += 5.0 * abs(float(err_a12))

    # Форма опасного режима.
    obj += 1.0 * safe_abs(error_row.get("err_max_alpha"))
    obj += 0.05 * safe_abs(error_row.get("err_altitude_loss_before_warning"))

    return float(obj)


# ============================================================
# Одна комбинация параметров
# ============================================================

def evaluate_landing_params(
    elevator_moment_gain,
    dCm0,
    dCL0=None,
    dCd0=None,
):
    """
    Проверяет одну комбинацию параметров LANDING.

    Возвращает одну строку для landing_calibration.csv.
    """

    landing_cfg = make_landing_config(
        elevator_moment_gain=elevator_moment_gain,
        dCm0=dCm0,
        dCL0=dCL0,
        dCd0=dCd0,
    )

    row = {
        "elevator_moment_gain": float(elevator_moment_gain),
        "dCm0": float(dCm0),
        "dCL0": safe_float(getattr(landing_cfg, "dCL0", np.nan)),
        "dCd0": safe_float(getattr(landing_cfg, "dCd0", np.nan)),
    }

    # ------------------------------------------------------------
    # 1. Own trim
    # ------------------------------------------------------------

    try:
        trim = find_own_model_trim(
            aircraft=BOEING_737_JSBSIM,
            config=landing_cfg,
            V=LANDING_TRIM_V,
            h=LANDING_TRIM_H,
            gamma_deg=LANDING_TRIM_GAMMA_DEG,
            name=LANDING_TRIM_NAME,
        )
    except Exception as exc:
        row.update({
            "status": "TRIM_ERROR",
            "error": str(exc),
            "objective": 1e9,
        })
        return row

    row.update({
        "trim_pass": bool(trim.get("trim_pass", False)),
        "trim_success": bool(trim.get("success", False)),
        "trim_cost": safe_float(trim.get("cost")),
        "trim_u": safe_float(trim.get("u")),
        "trim_theta_deg": safe_float(trim.get("theta_deg")),
        "trim_alpha_deg": safe_float(trim.get("alpha_deg")),
        "trim_throttle": safe_float(trim.get("throttle")),
        "trim_CL": safe_float(trim.get("CL")),
        "trim_CL_required": safe_float(trim.get("CL_required")),
        "trim_CL_error": safe_float(trim.get("CL_error")),
        "trim_dV": safe_float(trim.get("dV")),
        "trim_dgamma": safe_float(trim.get("dgamma")),
        "trim_dq": safe_float(trim.get("dq")),
        "trim_elevator_at_limit": bool(trim.get("elevator_at_limit", False)),
    })

    if not row["trim_pass"] or row["trim_elevator_at_limit"]:
        row.update({
            "status": "TRIM_REJECTED",
            "objective": 1e8 + safe_abs(row.get("trim_cost"), default=0.0),
        })
        return row

    # ------------------------------------------------------------
    # 2. Scenario sanity 5s
    # ------------------------------------------------------------

    try:
        base_scenario = get_scenario_by_name(LANDING_SCENARIO_NAME)
        scenario = make_calibration_scenario(
            base_scenario=base_scenario,
            landing_cfg=landing_cfg,
            trim=trim,
        )

        sanity_rows = run_own_scenario_short(scenario, t_final=5.0)
        sanity = summarize_rows(sanity_rows)

    except Exception as exc:
        row.update({
            "status": "SANITY_ERROR",
            "error": str(exc),
            "objective": 1e7,
        })
        return row

    row.update({
        "sanity_pass": bool(sanity.get("sanity_pass", False)),
        "sanity_V_change": safe_float(sanity.get("V_change")),
        "sanity_h_change": safe_float(sanity.get("h_change")),
        "sanity_h_change_error": safe_float(sanity.get("h_change_error")),
        "sanity_max_q": safe_float(sanity.get("max_abs_q")),
        "sanity_max_gamma_change_deg": safe_float(
            sanity.get("max_abs_gamma_change_deg")
        ),
        "sanity_max_theta_change_deg": safe_float(
            sanity.get("max_abs_theta_change_deg")
        ),
        "sanity_min_load_factor": safe_float(sanity.get("min_load_factor")),
        "sanity_max_R": safe_float(sanity.get("max_R")),
    })

    if not row["sanity_pass"]:
        row.update({
            "status": "SANITY_REJECTED",
            "objective": 1e6
            + 100.0 * safe_abs(row.get("sanity_V_change"), default=0.0)
            + 100.0 * safe_abs(row.get("sanity_max_q"), default=0.0),
        })
        return row

    # ------------------------------------------------------------
    # 3. Event metrics own vs JSBSim
    # ------------------------------------------------------------

    try:
        res_6dof = run_jsbsim_scenario(
            scenario,
            rotation_mode="6dof",
        )
        add_gamma_derivative_and_load_factor_estimate(res_6dof)

        res_own = run_own_model_with_replayed_controls(
            scenario,
            res_6dof,
        )
        add_gamma_derivative_and_load_factor_estimate(res_own)

        event_rows = [
            compute_event_metrics(res_own, scenario, "own"),
            compute_event_metrics(res_6dof, scenario, "jsbsim_6dof"),
        ]

        error_row = compute_event_error_row_local(event_rows, scenario)

    except Exception as exc:
        row.update({
            "status": "EVENT_ERROR",
            "error": str(exc),
            "objective": 1e5,
        })
        return row

    objective = compute_landing_objective(error_row)

    own_event = next(r for r in event_rows if r.get("model") == "own")
    jsb_event = next(r for r in event_rows if r.get("model") == "jsbsim_6dof")

    row.update({
        "status": "OK",
        "objective": objective,

        "own_phase": error_row.get("own_phase_reached"),
        "jsb_phase": error_row.get("jsb_phase_reached"),
        "same_phase": bool(error_row.get("same_phase_reached", False)),

        "own_warning": bool(error_row.get("own_warning_reached", False)),
        "jsb_warning": bool(error_row.get("jsb_warning_reached", False)),
        "own_stall": bool(error_row.get("own_stall_reached", False)),
        "jsb_stall": bool(error_row.get("jsb_stall_reached", False)),

        "own_t_alpha_8": own_event.get("t_alpha_8"),
        "jsb_t_alpha_8": jsb_event.get("t_alpha_8"),
        "err_t_alpha_8": error_row.get("err_t_alpha_8"),

        "own_t_alpha_10": own_event.get("t_alpha_10"),
        "jsb_t_alpha_10": jsb_event.get("t_alpha_10"),
        "err_t_alpha_10": error_row.get("err_t_alpha_10"),

        "own_t_alpha_12": own_event.get("t_alpha_12"),
        "jsb_t_alpha_12": jsb_event.get("t_alpha_12"),
        "err_t_alpha_12": error_row.get("err_t_alpha_12"),

        "own_t_warning_proxy": own_event.get("t_warning_proxy"),
        "jsb_t_warning_proxy": jsb_event.get("t_warning_proxy"),
        "err_t_warning_proxy": error_row.get("err_t_warning_proxy"),

        "own_t_stall_proxy": own_event.get("t_stall_proxy"),
        "jsb_t_stall_proxy": jsb_event.get("t_stall_proxy"),
        "err_t_stall_proxy": error_row.get("err_t_stall_proxy"),

        "own_max_alpha": own_event.get("max_alpha"),
        "jsb_max_alpha": jsb_event.get("max_alpha"),
        "err_max_alpha": error_row.get("err_max_alpha"),

        "own_min_V": own_event.get("min_V"),
        "jsb_min_V": jsb_event.get("min_V"),
        "err_min_V": error_row.get("err_min_V"),

        "own_min_speed_margin": own_event.get("min_speed_margin"),
        "jsb_min_speed_margin": jsb_event.get("min_speed_margin"),
        "err_min_speed_margin": error_row.get("err_min_speed_margin"),

        "own_altitude_loss_before_warning": own_event.get(
            "altitude_loss_before_warning"
        ),
        "jsb_altitude_loss_before_warning": jsb_event.get(
            "altitude_loss_before_warning"
        ),
        "err_altitude_loss_before_warning": error_row.get(
            "err_altitude_loss_before_warning"
        ),

        "mean_abs_early_event_error_s": error_row.get(
            "mean_abs_early_event_error_s"
        ),
        "max_abs_early_event_error_s": error_row.get(
            "max_abs_early_event_error_s"
        ),
    })

    return row


# ============================================================
# Полная калибровка
# ============================================================

def run_landing_calibration():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    if USE_EXTENDED_CL_CD_GRID:
        total = (
            len(ELEVATOR_GAIN_GRID)
            * len(DCM0_GRID)
            * len(DCL0_GRID)
            * len(DCD0_GRID)
        )
    else:
        total = len(ELEVATOR_GAIN_GRID) * len(DCM0_GRID)

    counter = 0

    print("\n=== LANDING calibration ===")
    print(f"Output: {OUT_CSV}")
    print(f"USE_EXTENDED_CL_CD_GRID = {USE_EXTENDED_CL_CD_GRID}")
    print(f"Total combinations: {total}")

    if USE_EXTENDED_CL_CD_GRID:
        for gain in ELEVATOR_GAIN_GRID:
            for dCm0 in DCM0_GRID:
                for dCL0 in DCL0_GRID:
                    for dCd0 in DCD0_GRID:
                        counter += 1

                        print(
                            f"\n[{counter}/{total}] "
                            f"gain={gain:.3f}, "
                            f"dCm0={dCm0:+.3f}, "
                            f"dCL0={dCL0:+.3f}, "
                            f"dCd0={dCd0:+.3f}"
                        )

                        row = evaluate_landing_params(
                            elevator_moment_gain=gain,
                            dCm0=dCm0,
                            dCL0=dCL0,
                            dCd0=dCd0,
                        )

                        rows.append(row)
                        save_rows_csv(OUT_CSV, rows)

                        print_result_line(row)

    else:
        for gain in ELEVATOR_GAIN_GRID:
            for dCm0 in DCM0_GRID:
                counter += 1

                print(
                    f"\n[{counter}/{total}] "
                    f"gain={gain:.3f}, "
                    f"dCm0={dCm0:+.3f}"
                )

                row = evaluate_landing_params(
                    elevator_moment_gain=gain,
                    dCm0=dCm0,
                )

                rows.append(row)
                save_rows_csv(OUT_CSV, rows)

                print_result_line(row)

    print("\n=== Calibration finished ===")
    print(f"Saved: {OUT_CSV}")

    print_best_rows(rows, n=10)

    return rows


def print_result_line(row):
    status = row.get("status")
    objective = row.get("objective")

    print(
        f"status={status} | "
        f"objective={objective} | "
        f"trim={row.get('trim_pass')} | "
        f"sanity={row.get('sanity_pass')} | "
        f"phase={row.get('own_phase')} vs {row.get('jsb_phase')} | "
        f"own_stall={row.get('own_stall')} | "
        f"jsb_stall={row.get('jsb_stall')} | "
        f"err_warn={row.get('err_t_warning_proxy')} | "
        f"err_alpha10={row.get('err_t_alpha_10')} | "
        f"err_max_alpha={row.get('err_max_alpha')}"
    )


def print_best_rows(rows, n=10):
    ok_rows = [
        r for r in rows
        if r.get("status") == "OK"
        and np.isfinite(safe_float(r.get("objective")))
    ]

    if not ok_rows:
        print("\nNo OK rows.")
        return

    ok_rows = sorted(ok_rows, key=lambda r: safe_float(r.get("objective")))

    print(f"\n=== Best {min(n, len(ok_rows))} LANDING configurations ===")

    for i, r in enumerate(ok_rows[:n], start=1):
        print(
            f"{i:02d}. "
            f"objective={r.get('objective'):.3f} | "
            f"gain={r.get('elevator_moment_gain')} | "
            f"dCm0={r.get('dCm0')} | "
            f"dCL0={r.get('dCL0')} | "
            f"dCd0={r.get('dCd0')} | "
            f"phase={r.get('own_phase')} vs {r.get('jsb_phase')} | "
            f"own_stall={r.get('own_stall')} | "
            f"jsb_stall={r.get('jsb_stall')} | "
            f"err_warn={r.get('err_t_warning_proxy')} | "
            f"err_a10={r.get('err_t_alpha_10')} | "
            f"err_a12={r.get('err_t_alpha_12')} | "
            f"err_max_alpha={r.get('err_max_alpha')}"
        )


def main():
    run_landing_calibration()


if __name__ == "__main__":
    main()