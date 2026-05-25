# validation/stall_event_metrics.py

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
from experiments.scenarios_737 import get_primary_validation_scenarios


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "stall_event_metrics")


def state_to_row(t, scenario, state, u, throttle):
    alpha_deg = np.degrees(state.alpha)
    theta_deg = np.degrees(state.theta)
    gamma_deg = np.degrees(state.gamma)

    return {
        "scenario": scenario.name,
        "group": scenario.group,
        "purpose": scenario.purpose,
        "parent_trim": scenario.parent_trim,
        "t": t,

        "u": float(u),
        "throttle": float(throttle),

        "V": float(state.V),
        "h": float(state.h),
        "x": float(getattr(state, "x", 0.0)),

        "theta_deg": float(theta_deg),
        "gamma_deg": float(gamma_deg),
        "alpha_deg": float(alpha_deg),
        "q": float(state.q),

        "CL": float(getattr(state, "CL", np.nan)),
        "CD": float(getattr(state, "CD", np.nan)),
        "Cm": float(getattr(state, "Cm", np.nan)),
        "sep": float(getattr(state, "sep", np.nan)),
        "R": float(getattr(state, "R", np.nan)),
        "mode": getattr(state, "mode", "UNKNOWN"),

        "load_factor": float(getattr(state, "load_factor", np.nan)),
        "V_stall_1g": float(getattr(state, "V_stall_1g", np.nan)),
        "V_stall_maneuver": float(getattr(state, "V_stall_maneuver", np.nan)),
        "speed_margin_1g": float(getattr(state, "speed_margin_1g", np.nan)),
        "speed_margin_maneuver": float(getattr(state, "speed_margin_maneuver", np.nan)),

        "vertical_speed": float(getattr(state, "vertical_speed", np.nan)),
        "energy_margin": float(getattr(state, "energy_margin", np.nan)),
        "stall_warning": bool(getattr(state, "stall_warning", False)),
        "stall_warning_margin": float(getattr(state, "stall_warning_margin", np.nan)),

        "elevator_eff": float(getattr(state, "elevator_eff", np.nan)),
        "elevator_speed_gain": float(getattr(state, "elevator_speed_gain", np.nan)),
        "elevator_config_gain": float(getattr(state, "elevator_config_gain", np.nan)),
        "Cm_delta_eff": float(getattr(state, "Cm_delta_eff", np.nan)),
    }


def run_own_scenario_full(scenario):
    aircraft = BOEING_737_JSBSIM
    dynamics = AircraftDynamics(aircraft, scenario.config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    state = scenario.make_initial_state()

    rows = []
    t = 0.0
    n_steps = int(np.ceil(scenario.t_final / dt))

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

        if abs(state.theta) > np.radians(75.0) or abs(state.gamma) > np.radians(75.0):
            state.mode = "OUT_OF_ENVELOPE"
            rows.append(state_to_row(t, scenario, state, u, throttle))
            break

        t += dt

    return rows


def first_time(rows, condition):
    for row in rows:
        if condition(row):
            return row["t"]
    return None


def row_at_time(rows, t_event):
    if t_event is None:
        return None
    return min(rows, key=lambda r: abs(r["t"] - t_event))


def altitude_metrics(rows, t_event):
    """
    Возвращает:
    - altitude_change_from_start: h_event - h_start
    - altitude_loss_from_start: max(0, h_start - h_event)
    - altitude_loss_from_peak: max_h_until_event - h_event

    Для climb-сценариев особенно полезен altitude_loss_from_peak.
    """
    if t_event is None:
        return {
            "altitude_change_from_start": None,
            "altitude_loss_from_start": None,
            "altitude_loss_from_peak": None,
        }

    h0 = rows[0]["h"]
    event_row = row_at_time(rows, t_event)
    h_event = event_row["h"]

    rows_until_event = [r for r in rows if r["t"] <= t_event]
    h_peak = max(r["h"] for r in rows_until_event)

    return {
        "altitude_change_from_start": h_event - h0,
        "altitude_loss_from_start": max(0.0, h0 - h_event),
        "altitude_loss_from_peak": max(0.0, h_peak - h_event),
    }


def summarize_events(rows):
    if not rows:
        return {}

    scenario = rows[0]["scenario"]
    group = rows[0]["group"]
    purpose = rows[0]["purpose"]
    parent_trim = rows[0]["parent_trim"]

    t_alpha_8 = first_time(rows, lambda r: r["alpha_deg"] >= 8.0)
    t_alpha_10 = first_time(rows, lambda r: r["alpha_deg"] >= 10.0)
    t_alpha_12 = first_time(rows, lambda r: r["alpha_deg"] >= 12.0)

    t_warning = first_time(
        rows,
        lambda r: bool(r["stall_warning"]) or r["mode"] in ["WARNING", "STALL"]
    )

    t_stall = first_time(
        rows,
        lambda r: r["mode"] == "STALL" or r["sep"] >= 0.7
    )

    min_speed_margin_1g = min(r["speed_margin_1g"] for r in rows)
    min_speed_margin_maneuver = min(r["speed_margin_maneuver"] for r in rows)

    max_alpha = max(r["alpha_deg"] for r in rows)
    max_R = max(r["R"] for r in rows)
    max_sep = max(r["sep"] for r in rows)
    min_V = min(r["V"] for r in rows)
    min_h = min(r["h"] for r in rows)

    warning_alt = altitude_metrics(rows, t_warning)
    stall_alt = altitude_metrics(rows, t_stall)

    final = rows[-1]

    return {
        "scenario": scenario,
        "group": group,
        "purpose": purpose,
        "parent_trim": parent_trim,

        "t_final_reached": final["t"],
        "final_mode": final["mode"],

        "t_alpha_8": t_alpha_8,
        "t_alpha_10": t_alpha_10,
        "t_alpha_12": t_alpha_12,
        "t_warning": t_warning,
        "t_stall": t_stall,

        "warning_reached": t_warning is not None,
        "stall_reached": t_stall is not None,

        "min_speed_margin_1g": min_speed_margin_1g,
        "min_speed_margin_maneuver": min_speed_margin_maneuver,
        "max_alpha_deg": max_alpha,
        "max_R": max_R,
        "max_sep": max_sep,
        "min_V": min_V,
        "min_h": min_h,

        "altitude_change_before_warning": warning_alt["altitude_change_from_start"],
        "altitude_loss_before_warning": warning_alt["altitude_loss_from_start"],
        "altitude_loss_from_peak_before_warning": warning_alt["altitude_loss_from_peak"],

        "altitude_change_before_stall": stall_alt["altitude_change_from_start"],
        "altitude_loss_before_stall": stall_alt["altitude_loss_from_start"],
        "altitude_loss_from_peak_before_stall": stall_alt["altitude_loss_from_peak"],
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

    scenarios = [
        s for s in get_primary_validation_scenarios()
        if s.group == "stall_entry"
    ]

    all_rows = []
    summary_rows = []

    for scenario in scenarios:
        print(f"\n=== Running stall-entry scenario: {scenario.name} ===")

        rows = run_own_scenario_full(scenario)
        summary = summarize_events(rows)

        scenario_path = os.path.join(RESULTS_DIR, f"{scenario.name}_trajectory.csv")
        save_csv(scenario_path, rows)

        all_rows.extend(rows)
        summary_rows.append(summary)

        print(
            f"warning={summary['warning_reached']} "
            f"t_warning={summary['t_warning']} | "
            f"stall={summary['stall_reached']} "
            f"t_stall={summary['t_stall']} | "
            f"max_alpha={summary['max_alpha_deg']:.2f} deg | "
            f"min_margin={summary['min_speed_margin_maneuver']:.3f} | "
            f"max_R={summary['max_R']:.3f}"
        )

    save_csv(os.path.join(RESULTS_DIR, "all_stall_entry_trajectories.csv"), all_rows)
    save_csv(os.path.join(RESULTS_DIR, "stall_event_summary.csv"), summary_rows)

    print(f"\nSaved stall event metrics to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
