# experiments/run_experiments.py

import os
import csv
import numpy as np
import random
import pandas as pd
import copy

from config import dt
from AircraftModel import BOEING_737_800
from model.dynamics import AircraftDynamics
from model.risk import RiskModel
from simulation.rk2 import RK2Integrator
from experiments.scenarios import SCENARIOS
from experiments.scenarios_737 import SCENARIOS_737
from AircraftModel import BOEING_737_800, CLEAN, LANDING

N_RUNS_PER_VARIANT = 3
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
RESULTS_DIR = "results"
SENSITIVITY_SCENARIO_NAMES = [
    "energy_fight_climb",
    "pitch_fight_instability",
    "slow_climb_control_loss",
    "turbulence_induced_stall",
    "flare_overcontrol",
    "normal_flight_clean",
    "normal_climb_clean",
]

def state_to_row(t, scenario, state, u=None, throttle_cmd=None):
    return {
        "scenario": scenario.name,
        "run_id": getattr(state, "run_id", None),
        "seed": getattr(state, "seed", None),
        "parameter_name": getattr(state, "parameter_name", None),
        "parameter_variant": getattr(state, "parameter_variant", None),
        "parameter_value": getattr(state, "parameter_value", None),
        "description": scenario.description,
        "config": scenario.config.name,
        "t": t,
        "theta_rad": state.theta,
        "theta_deg": np.degrees(state.theta),
        "gamma_rad": state.gamma,
        "gamma_deg": np.degrees(state.gamma),
        "alpha_rad": state.alpha,
        "alpha_deg": np.degrees(state.alpha),
        "q": state.q,
        "V": state.V,
        "h": state.h,
        "x": state.x,
        "vertical_speed": state.vertical_speed,
        "CL": state.CL,
        "CD": state.CD,
        "Cm": state.Cm,
        "CL_ratio": state.CL_ratio,
        "sep": state.sep,
        "V_stall_1g": state.V_stall_1g,
        "V_stall_maneuver": state.V_stall_maneuver,
        "speed_margin_1g": state.speed_margin_1g,
        "speed_margin_maneuver": state.speed_margin_maneuver,
        "load_factor": state.load_factor,
        "throttle": state.throttle,
        "thrust": state.thrust,
        "power_available": state.power_available,
        "power_required_total": state.power_required_total,
        "excess_power": state.excess_power,
        "energy_margin": state.energy_margin,
        "R": state.R,
        "mode": state.mode,
        "throttle_command": throttle_cmd,
        "stall_warning": getattr(state, "stall_warning", False),
        "stall_warning_speed": getattr(state, "stall_warning_speed", 0.0),
        "stall_warning_margin": getattr(state, "stall_warning_margin", 0.0),
        "elevator_command": u,
        "is_stall": int(state.mode == "STALL"),
        "is_warning": int(state.mode == "WARNING"),
        "is_critical": int(state.sep > 0.7),

    }


def run_scenario(scenario, aircraft, run_id=None, seed=None,
                  parameter_name=None, parameter_variant=None, parameter_value=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    dynamics = AircraftDynamics(aircraft, scenario.config)
    risk_model = RiskModel()
    integrator = RK2Integrator()

    state = scenario.make_initial_state()
    state.run_id = run_id
    state.seed = seed
    state.parameter_name = parameter_name
    state.parameter_variant = parameter_variant
    state.parameter_value = parameter_value

    t = 0.0
    rows = []

    n_steps = int(np.ceil(scenario.t_final / dt))

    for _ in range(n_steps):
        if t > scenario.t_final:
            break
        u, throttle = scenario.control_law(t, state)
        state.elevator_command = u

        state = integrator.step(
            state=state,
            dynamics=dynamics,
            risk_model=risk_model,
            u=u,
            throttle=throttle,
            dt=dt,
        )



        if abs(state.theta) > np.radians(55.0) or abs(state.gamma) > np.radians(50.0):
            state.mode = "OUT_OF_ENVELOPE"
            rows.append(state_to_row(t, scenario, state, u, throttle))
            break


        rows.append(state_to_row(t, scenario, state, u, throttle))


        if state.V < 5.0 or state.h < 0.0:
            break

        t += dt

    return rows

# def run_scenario_with_aircraft(scenario, aircraft):
#     dynamics = AircraftDynamics(aircraft, scenario.config)
#     risk_model = RiskModel()
#     integrator = RK2Integrator()
#
#     state = scenario.make_initial_state()
#
#     t = 0.0
#     rows = []
#
#     n_steps = int(np.ceil(scenario.t_final / dt))
#
#     for _ in range(n_steps):
#         u, throttle = scenario.control_law(t, state)
#
#         state = integrator.step(
#             state=state,
#             dynamics=dynamics,
#             risk_model=risk_model,
#             u=u,
#             throttle=throttle,
#             dt=dt,
#         )
#
#         rows.append(state_to_row(t, scenario, state))
#
#         if state.V < 5.0 or state.h < 0.0:
#             break
#
#         if abs(state.theta) > np.radians(55.0) or abs(state.gamma) > np.radians(50.0):
#             state.mode = "OUT_OF_ENVELOPE"
#             rows.append(state_to_row(t, scenario, state))
#             break
#
#         t += dt
#
#     return rows


def summarize_sensitivity(rows):
    if not rows:
        return {}

    t0 = rows[0]["t"]
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
    """
    Возвращает список вариантов:
    parameter_name, variant_label, new_value
    """

    grid = []

    # 1. alpha_stall_on — варьируем в градусах
    alpha_base_deg = np.degrees(base_aircraft.alpha_stall_on)
    for delta_deg in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        new_alpha = np.radians(alpha_base_deg + delta_deg)
        grid.append((
            "alpha_stall_on",
            f"{delta_deg:+.0f}deg",
            new_alpha
        ))

    # 2. Остальные параметры — варьируем множителями
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

def summarize(rows):
    if not rows:
        return {}

    max_R = max(row["R"] for row in rows)
    max_sep = max(row["sep"] for row in rows)
    min_margin_1g = min(row["speed_margin_1g"] for row in rows)
    min_margin_maneuver = min(row["speed_margin_maneuver"] for row in rows)
    min_V = min(row["V"] for row in rows)
    max_alpha = max(abs(row["alpha_deg"]) for row in rows)
    final_mode = rows[-1]["mode"]

    stall_warning_triggered = any(row["stall_warning"] for row in rows)
    stall_reached = any(row["mode"] == "STALL" or row["sep"] > 0.7 for row in rows)
    low_speed_event = any(row["speed_margin_maneuver"] < 1.0 for row in rows)

    t_first_warning = first_time(rows, lambda r: r["stall_warning"])
    t_first_low_margin = first_time(rows, lambda r: r["speed_margin_maneuver"] < 1.0)
    t_first_stall = first_time(rows, lambda r: r["mode"] == "STALL" or r["sep"] > 0.7)

    return {
        "max_R": max_R,
        "max_sep": max_sep,
        "min_margin_1g": min_margin_1g,
        "min_margin_maneuver": min_margin_maneuver,
        "min_V": min_V,
        "max_abs_alpha_deg": max_alpha,
        "final_mode": final_mode,
        "stall_warning_triggered": stall_warning_triggered,
        "low_speed_event": low_speed_event,
        "stall_reached": stall_reached,
        "t_first_warning": t_first_warning,
        "t_first_low_margin": t_first_low_margin,
        "t_first_stall": t_first_stall,
    }

def first_time(rows, condition):
    for row in rows:
        if condition(row):
            return row["t"]
    return None
def save_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []
    # all_rows = []
    all_path = os.path.join(RESULTS_DIR, "all_runs_ml_ready.csv")
    all_file = open(all_path, "w", newline="", encoding="utf-8")
    writer = None

    for scenario in SCENARIOS_737:
        print(f"Running scenario: {scenario.name}")


        grid = build_parameter_grid(BOEING_737_800)
        for parameter_name, parameter_variant, parameter_value in grid:

            for i in range(N_RUNS_PER_VARIANT):
                seed = GLOBAL_SEED + hash((scenario.name, parameter_name, parameter_variant, i)) % 10**6
                run_id = f"{scenario.name}_{parameter_name}_{parameter_variant}_{i}"

                aircraft = copy.deepcopy(BOEING_737_800)
                if not hasattr(aircraft, parameter_name):
                    raise ValueError(f"Aircraft has no parameter {parameter_name}")

                setattr(aircraft, parameter_name, parameter_value)

                rows = run_scenario(
                    scenario,
                    aircraft=aircraft,
                    run_id=run_id,
                    seed=seed,
                    parameter_name=parameter_name,
                    parameter_variant=parameter_variant,
                    parameter_value=parameter_value
                )


                if len(rows) > 1:
                    if all(r["CL"] == rows[0]["CL"] for r in rows[:5]):
                        print(f"WARNING: {parameter_name} does not affect dynamics")
                csv_path = os.path.join(RESULTS_DIR, f"{run_id}.csv")
                save_csv(csv_path, rows)
                if rows:
                    if writer is None:
                        fieldnames = list(rows[0].keys())
                        writer = csv.DictWriter(all_file, fieldnames=fieldnames)
                        writer.writeheader()

                    writer.writerows(rows)

                summary = summarize(rows)
                summary["scenario"] = scenario.name
                summary["description"] = scenario.description
                summary.update({
                    "run_id": run_id,
                    "seed": seed,
                    "parameter_name": parameter_name,
                    "parameter_variant": parameter_variant,
                    "parameter_value": parameter_value,
                })
                summary_rows.append(summary)
                #all_rows.extend(rows)

                print(
                    f"  max_R={summary['max_R']:.2f}, "
                    f"max_sep={summary['max_sep']:.2f}, "
                    f"min_margin={summary['min_margin_maneuver']:.2f}, "
                    f"stall={summary['stall_reached']}"
                )
    # df_all = pd.DataFrame(all_rows)
    # df_all.to_csv(os.path.join(RESULTS_DIR, "all_runs_ml_ready.csv"), index=False)
    # df_all.to_parquet(os.path.join(RESULTS_DIR, "all_runs_ml_ready.parquet"))
    #
    # summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    # save_csv(summary_path, summary_rows)
    # all_path = os.path.join(RESULTS_DIR, "all_scenarios.csv")
    # save_csv(all_path, all_rows)

    all_file.close()

    # --- summary ---
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    save_csv(summary_path, summary_rows)

    print(f"\nSaved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()