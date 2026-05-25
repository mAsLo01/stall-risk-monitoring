# experiments/sensitivity.py

import os
import copy
import numpy as np

from config import dt
from AircraftModel import CESSNA_172
from model.dynamics import AircraftDynamics
from model.risk import RiskModel
from simulation.rk2 import RK2Integrator
from experiments.scenarios import SCENARIOS
from experiments.run_experiments import state_to_row, save_csv


RESULTS_DIR = "results/sensitivity"

SENSITIVITY_SCENARIO_NAMES = [
    "clean_power_off_stall",
    "clean_power_on_stall",
    "landing_config_stall",
    "stall_recovery_clean",
    "accelerated_pull_up",
]


def clone_aircraft_with_change(base_aircraft, parameter_name, new_value):
    aircraft = copy.deepcopy(base_aircraft)
    setattr(aircraft, parameter_name, new_value)

    # Пересчёт зависимых параметров:
    # AR, k, CL_max_clean, CL_alpha, CL_plateau и т.п.
    aircraft.__post_init__()

    return aircraft


def run_scenario_with_aircraft(scenario, aircraft):
    dynamics = AircraftDynamics(aircraft, scenario.config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    state = scenario.make_initial_state()

    t = 0.0
    rows = []

    n_steps = int(np.ceil(scenario.t_final / dt))

    for _ in range(n_steps):
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

        if abs(state.theta) > np.radians(55.0) or abs(state.gamma) > np.radians(50.0):
            state.mode = "OUT_OF_ENVELOPE"
            rows.append(state_to_row(t, scenario.name, state))
            break

        t += dt

    return rows


def summarize_sensitivity(rows):
    if not rows:
        return {}

    h0 = rows[0]["h"]

    max_R = max(row["R"] for row in rows)
    max_sep = max(row["sep"] for row in rows)
    min_margin_1g = min(row["speed_margin_1g"] for row in rows)
    min_margin_maneuver = min(row["speed_margin_maneuver"] for row in rows)
    min_V = min(row["V"] for row in rows)
    min_h = min(row["h"] for row in rows)
    max_alpha = max(abs(row["alpha_deg"]) for row in rows)
    final_mode = rows[-1]["mode"]

    stall_rows = [row for row in rows if row["mode"] == "STALL"]
    warning_rows = [row for row in rows if row["mode"] in ["WARNING", "STALL"]]

    time_to_stall = stall_rows[0]["t"] if stall_rows else None
    time_to_warning = warning_rows[0]["t"] if warning_rows else None

    stall_reached = len(stall_rows) > 0
    warning_reached = len(warning_rows) > 0

    altitude_loss = h0 - min_h

    return {
        "max_R": max_R,
        "max_sep": max_sep,
        "min_margin_1g": min_margin_1g,
        "min_margin_maneuver": min_margin_maneuver,
        "min_V": min_V,
        "min_h": min_h,
        "altitude_loss": altitude_loss,
        "max_abs_alpha_deg": max_alpha,
        "final_mode": final_mode,
        "warning_reached": warning_reached,
        "stall_reached": stall_reached,
        "time_to_warning": time_to_warning,
        "time_to_stall": time_to_stall,
    }


def build_parameter_grid(base_aircraft):
    grid = []

    alpha_base_deg = np.degrees(base_aircraft.alpha_stall_on)

    for delta_deg in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        new_alpha = np.radians(alpha_base_deg + delta_deg)
        grid.append((
            "alpha_stall_on",
            f"{delta_deg:+.0f}deg",
            new_alpha
        ))

    relative_parameters = [
        "tau_sep_stall",
        "tau_sep_recover",
        "CL_plateau_ratio",
        "Cm_alpha",
        "Cm_delta_e",
    ]

    multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]

    for parameter_name in relative_parameters:
        base_value = getattr(base_aircraft, parameter_name)

        for multiplier in multipliers:
            new_value = base_value * multiplier
            grid.append((
                parameter_name,
                f"x{multiplier:.1f}",
                new_value
            ))

    return grid


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scenarios = [
        scenario for scenario in SCENARIOS
        if scenario.name in SENSITIVITY_SCENARIO_NAMES
    ]

    parameter_grid = build_parameter_grid(CESSNA_172)

    summary_rows = []

    for scenario in scenarios:
        scenario_dir = os.path.join(RESULTS_DIR, scenario.name)
        os.makedirs(scenario_dir, exist_ok=True)

        for parameter_name, variant_label, new_value in parameter_grid:
            aircraft = clone_aircraft_with_change(
                CESSNA_172,
                parameter_name,
                new_value
            )

            rows = run_scenario_with_aircraft(scenario, aircraft)

            safe_value = (
                variant_label
                .replace("+", "plus")
                .replace("-", "minus")
                .replace(".", "p")
            )

            csv_name = f"{scenario.name}__{parameter_name}__{safe_value}.csv"
            csv_path = os.path.join(scenario_dir, csv_name)

            save_csv(csv_path, rows)

            summary = summarize_sensitivity(rows)

            summary["scenario"] = scenario.name
            summary["parameter"] = parameter_name
            summary["variant"] = variant_label
            summary["value"] = new_value

            if parameter_name == "alpha_stall_on":
                summary["value_human"] = np.degrees(new_value)
                summary["unit"] = "deg"
            else:
                summary["value_human"] = new_value
                summary["unit"] = "-"

            summary_rows.append(summary)

            print(
                f"{scenario.name:28s} | "
                f"{parameter_name:18s} | "
                f"{variant_label:8s} | "
                f"stall={summary['stall_reached']} | "
                f"max_R={summary['max_R']:.2f} | "
                f"min_margin={summary['min_margin_maneuver']:.2f}"
            )

    summary_path = os.path.join(RESULTS_DIR, "sensitivity_summary.csv")
    save_csv(summary_path, summary_rows)

    print(f"\nSaved sensitivity results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()