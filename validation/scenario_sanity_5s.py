# validation/scenario_sanity_5s.py

import os
import sys
import csv
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import dt
from AircraftModel import BOEING_737_JSBSIM
from model.dynamics import AircraftDynamics
from model.risk import RiskModel
from simulation.rk2 import RK2Integrator
from experiments.scenarios_737 import (
    get_scenarios_by_group,
    get_primary_validation_scenarios,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "scenario_sanity")


def state_to_row(t, scenario, state, u, throttle):
    return {
        "scenario": scenario.name,
        "group": scenario.group,
        "parent_trim": scenario.parent_trim,
        "t": t,

        "u": u,
        "throttle": throttle,

        "V": state.V,
        "h": state.h,
        "theta_deg": np.degrees(state.theta),
        "gamma_deg": np.degrees(state.gamma),
        "alpha_deg": np.degrees(state.alpha),
        "q": state.q,

        "CL": getattr(state, "CL", np.nan),
        "CD": getattr(state, "CD", np.nan),
        "Cm": getattr(state, "Cm", np.nan),
        "sep": getattr(state, "sep", np.nan),
        "R": getattr(state, "R", np.nan),
        "mode": getattr(state, "mode", "UNKNOWN"),

        "load_factor": getattr(state, "load_factor", np.nan),
        "speed_margin_1g": getattr(state, "speed_margin_1g", np.nan),
        "speed_margin_maneuver": getattr(state, "speed_margin_maneuver", np.nan),
        "vertical_speed": getattr(state, "vertical_speed", np.nan),
        "energy_margin": getattr(state, "energy_margin", np.nan),

        "stall_warning": getattr(state, "stall_warning", False),
        "elevator_eff": getattr(state, "elevator_eff", np.nan),
        "elevator_speed_gain": getattr(state, "elevator_speed_gain", np.nan),
        "elevator_config_gain": getattr(state, "elevator_config_gain", np.nan),
        "Cm_delta_eff": getattr(state, "Cm_delta_eff", np.nan),
    }


def run_own_scenario_short(scenario, t_final=5.0):
    aircraft = BOEING_737_JSBSIM
    dynamics = AircraftDynamics(aircraft, scenario.config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    state = scenario.make_initial_state()

    rows = []
    t = 0.0
    n_steps = int(np.ceil(t_final / dt))

    for _ in range(n_steps + 1):
        u, throttle = scenario.control_law(t, state)

        state = integrator.step(
            state=state,
            dynamics=dynamics,
            risk_model=risk_model,
            u=u,
            throttle=throttle,
            dt=dt,
        )

        rows.append(state_to_row(t, scenario, state, u, throttle))

        if state.V < 5.0 or state.h < 0.0:
            break

        if abs(state.theta) > np.radians(60.0) or abs(state.gamma) > np.radians(60.0):
            state.mode = "OUT_OF_ENVELOPE"
            break

        t += dt

    return rows


def summarize_rows(rows):
    if not rows:
        return {}

    first = rows[0]
    last = rows[-1]

    max_abs_q = max(abs(r["q"]) for r in rows)
    max_abs_gamma_change = max(abs(r["gamma_deg"] - first["gamma_deg"]) for r in rows)
    max_abs_theta_change = max(abs(r["theta_deg"] - first["theta_deg"]) for r in rows)
    max_abs_alpha_change = max(abs(r["alpha_deg"] - first["alpha_deg"]) for r in rows)

    min_speed_margin = min(r["speed_margin_maneuver"] for r in rows)
    min_load_factor = min(r["load_factor"] for r in rows)
    max_R = max(r["R"] for r in rows)

    V_change = last["V"] - first["V"]
    h_change = last["h"] - first["h"]
    duration = last["t"] - first["t"]
    expected_h_change = (
            first["V"]
            * np.sin(np.radians(first["gamma_deg"]))
            * duration
    )

    h_change_error = h_change - expected_h_change

    failed = (
        abs(V_change) > 2.0
        or abs(h_change_error) > 10.0
        or max_abs_gamma_change > 3.0
        or max_abs_theta_change > 3.0
        or max_abs_q > 0.05
        or min_load_factor < 0.7
        or last["mode"] == "OUT_OF_ENVELOPE"
    )

    return {
        "scenario": first["scenario"],
        "group": first["group"],
        "parent_trim": first["parent_trim"],

        "t_start": first["t"],
        "t_end": last["t"],

        "V_start": first["V"],
        "V_end": last["V"],
        "V_change": V_change,

        "h_start": first["h"],
        "h_end": last["h"],
        "h_change": h_change,

        "theta_start_deg": first["theta_deg"],
        "theta_end_deg": last["theta_deg"],
        "gamma_start_deg": first["gamma_deg"],
        "gamma_end_deg": last["gamma_deg"],
        "alpha_start_deg": first["alpha_deg"],
        "alpha_end_deg": last["alpha_deg"],

        "max_abs_q": max_abs_q,
        "max_abs_gamma_change_deg": max_abs_gamma_change,
        "max_abs_theta_change_deg": max_abs_theta_change,
        "max_abs_alpha_change_deg": max_abs_alpha_change,

        "min_speed_margin_maneuver": min_speed_margin,
        "min_load_factor": min_load_factor,
        "max_R": max_R,
        "final_mode": last["mode"],
        "expected_h_change": expected_h_change,
        "h_change_error": h_change_error,

        "sanity_pass": not failed,
    }


def save_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Все trim-сценарии группы 2.
    trim_scenarios = get_scenarios_by_group("validation_trim")

    # 2. Основные сценарии группы 3, но проверяем только первые 5 секунд.
    # Важно: в самих сценариях первые 5 секунд должны совпадать с parent trim.
    stall_scenarios = [
        s for s in get_primary_validation_scenarios()
        if s.group == "stall_entry"
    ]

    scenarios = trim_scenarios + stall_scenarios

    all_rows = []
    summary_rows = []

    for scenario in scenarios:
        print(f"\n=== Sanity check: {scenario.name} ===")

        rows = run_own_scenario_short(scenario, t_final=5.0)
        summary = summarize_rows(rows)

        print(
            f"pass={summary['sanity_pass']} | "
            f"dV={summary['V_change']:+.3f} m/s | "
            f"dh={summary['h_change']:+.3f} m | "
            f"dh_expected={summary['expected_h_change']:+.3f} m | "
            f"dh_err={summary['h_change_error']:+.3f} m | "
            f"dtheta={summary['theta_end_deg'] - summary['theta_start_deg']:+.3f} deg | "
            f"dgamma={summary['gamma_end_deg'] - summary['gamma_start_deg']:+.3f} deg | "
            f"max_q={summary['max_abs_q']:.4f}"
        )

        scenario_path = os.path.join(RESULTS_DIR, f"{scenario.name}_5s.csv")
        save_csv(scenario_path, rows)

        all_rows.extend(rows)
        summary_rows.append(summary)

    save_csv(os.path.join(RESULTS_DIR, "all_sanity_5s.csv"), all_rows)
    save_csv(os.path.join(RESULTS_DIR, "summary_sanity_5s.csv"), summary_rows)

    print(f"\nSaved sanity results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()