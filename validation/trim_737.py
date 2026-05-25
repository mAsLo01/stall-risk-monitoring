from dataclasses import dataclass
import numpy as np
import os
import csv
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scipy.optimize import minimize

from AircraftModel import BOEING_737_JSBSIM, CLEAN, LANDING
from model.dynamics import AircraftDynamics
from atmosphere import Atmosphere
from config import dt as DT



from analytic_validation import (
    make_tmp_state,
    unpack_derivatives,
    find_u_for_Cm_zero,
)

from jsbsim_static_identification import (
    make_jsbsim_instance,
    get_jsbsim_state,
    finite_difference,
    apply_jsbsim_controls,
    safe_get,
    set_if_exists,
)


G = 9.81

@dataclass
class TrimTarget:
    name: str
    config: object
    V: float
    h: float
    gamma_deg: float
    description: str


TRIM_TARGETS_737 = [
    TrimTarget(
        name="clean_low_speed_trim_737",
        config=CLEAN,
        V=110.0,
        h=2000.0,
        gamma_deg=0.0,
        description="Clean low-speed level trim before stall-entry scenarios."
    ),
    TrimTarget(
        name="landing_low_speed_trim_737",
        config=LANDING,
        V=95.0,
        h=800.0,
        gamma_deg=0.0,
        description="Landing configuration low-speed level trim."
    ),
    TrimTarget(
        name="low_speed_climb_trim_737",
        config=CLEAN,
        V=115.0,
        h=1500.0,
        gamma_deg=3.0,
        description="Low-speed climb trim near energy-limited regime."
    ),
]

def trim_passed(dV, dgamma, dq):
    return (
        abs(dV) < 0.05
        and abs(dgamma) < 0.001
        and abs(dq) < 0.001
    )

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

        apply_jsbsim_controls(
            jsb,
            elevator_cmd_norm=0.0,
            throttle_cmd=throttle_cmd,
        )

        jsb.run()

def find_own_model_trim(aircraft, config, V, h, gamma_deg=0.0, name="own_trim"):
    """
    Подбирает trim для собственной reduced-order модели.

    Оптимизируются:
    - theta
    - throttle
    - u

    Критерий:
    - dV ≈ 0
    - dgamma ≈ 0
    - dq ≈ 0
    """

    dynamics = AircraftDynamics(aircraft, config)
    gamma = np.radians(gamma_deg)

    rho = Atmosphere().density(h)
    q_dyn = 0.5 * rho * V ** 2
    W = aircraft.mass * G

    CL_req = W * np.cos(gamma) / max(q_dyn * aircraft.wing_area, 1e-9)

    CL0 = aircraft.CL0 + config.dCL0
    alpha_est = (CL_req - CL0) / max(aircraft.CL_alpha, 1e-9)
    alpha_est = np.clip(alpha_est, np.radians(-5.0), np.radians(15.0))

    theta0 = alpha_est + gamma

    CD_est = aircraft.Cd0 + config.dCd0 + aircraft.k * CL_req ** 2
    D_est = q_dyn * aircraft.wing_area * CD_est
    throttle0 = np.clip(D_est / max(aircraft.static_thrust_max, 1e-9), 0.05, 0.95)

    u0 = find_u_for_Cm_zero(
        aircraft=aircraft,
        config=config,
        alpha=alpha_est,
        V=V,
        gamma=gamma,
        throttle=throttle0,
        h=h
    )

    x0 = np.array([theta0, throttle0, u0])

    bounds = [
        (np.radians(-5.0), np.radians(25.0)),  # theta
        (0.0, 1.0),  # throttle
        (-1.0, 1.0),  # u
    ]

    def cost(params):
        theta, throttle, u = params
        alpha = theta - gamma

        state = make_tmp_state(
            alpha=alpha,
            gamma=gamma,
            V=V,
            h=h
        )

        result = dynamics.derivatives(
            state,
            u=float(u),
            throttle=float(throttle),
            dt=DT
        )

        _, dq, dV, dgamma, _, _, _, _ = unpack_derivatives(result)

        return (
            (dV / 0.1) ** 2
            + (dgamma / 0.001) ** 2
            + (dq / 0.001) ** 2
        )


    res = minimize(
        cost,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "ftol": 1e-12,
            "gtol": 1e-12,
            "maxiter": 1000,
        }
    )

    theta_opt, throttle_opt, u_opt = res.x
    alpha_opt = theta_opt - gamma

    state_test = make_tmp_state(alpha_opt, gamma, V, h)

    result = dynamics.derivatives(
        state_test,
        u=float(u_opt),
        throttle=float(throttle_opt),
        dt=DT
    )


    _, dq, dV, dgamma, _, _, _, diagnostics = unpack_derivatives(result)

    aero, forces, stall, moment, energy, throttle_used = diagnostics

    W = aircraft.mass * G
    rho = Atmosphere().density(h)
    q_dyn = 0.5 * rho * V ** 2

    CL_required = W * np.cos(gamma) / max(q_dyn * aircraft.wing_area, 1e-9)

    trim_result = {
        "scenario_name": name,
        "backend": "own",
        "config": config.name,
        "V": V,
        "h": h,
        "gamma_deg": gamma_deg,
        "theta_deg": np.degrees(theta_opt),
        "alpha_deg": np.degrees(alpha_opt),
        "u": float(u_opt),
        "throttle": float(throttle_opt),
        "dV": float(dV),
        "dgamma": float(dgamma),
        "dq": float(dq),
        "CL": float(aero.CL),
        "CD": float(aero.CD),
        "Cm": float(moment.Cm),
        "load_factor": float(stall.load_factor),
        "success": bool(res.success),
        "cost": float(res.fun),
        "throttle_used": float(throttle_used),
        "trim_pass": trim_passed(dV, dgamma, dq),
        "CL_required": float(CL_required),
        "CL_error": float(aero.CL - CL_required),
        "delta_e_rad": float(u_opt * aircraft.max_elevator),
        "delta_e_deg": float(np.degrees(u_opt * aircraft.max_elevator)),
        "moment_M": float(moment.M),
        "elevator_at_limit": bool(abs(u_opt) > 0.999),
        "actual_V": float(V),
        "actual_h": float(h),
        "actual_theta_deg": float(np.degrees(theta_opt)),
        "actual_gamma_deg": float(gamma_deg),
        "actual_alpha_deg": float(np.degrees(alpha_opt)),
        "actual_q": 0.0,
    }

    return trim_result


def find_jsbsim_trim(
    target,
    theta0_deg=5.0,
    u0=0.0,
    throttle0=0.5,
    dt_local=0.001,
    rotation_mode="6dof",
):
    """
    Подбирает trim для JSBSim.

    Оптимизируются:
    - theta_deg
    - elevator command u
    - throttle

    Критерий:
    - dV ≈ 0
    - dgamma ≈ 0
    - dq ≈ 0
    """

    def evaluate(params):
        theta_deg, u, throttle = params

        gamma_deg = target.gamma_deg
        alpha_deg = theta_deg - gamma_deg

        jsb = make_jsbsim_instance(
            V_mps=target.V,
            h_m=target.h,
            alpha_deg=alpha_deg,
            gamma_deg=gamma_deg,
            dt_local=dt_local,
            rotation_mode=rotation_mode,
            q_rad_s=0.0,
        )

        # Важно: двигатели должны быть инициализированы так же,
        # как в твоём jsbsim_validation.py
        initialize_jsbsim_engines(jsb, throttle_cmd=float(throttle))

        # Важно: конфигурация CLEAN / LANDING должна применяться к JSBSim
        apply_jsbsim_configuration(jsb, target.config)

        # Дадим FCS и двигателю применить команды
        for _ in range(20):
            apply_jsbsim_controls(
                jsb,
                elevator_cmd_norm=float(u),
                throttle_cmd=float(throttle)
            )
            jsb.run()

        before = get_jsbsim_state(jsb)

        actual_elevator_rad = safe_get(jsb, "fcs/elevator-pos-rad", np.nan)
        thrust_0_lbs = safe_get(jsb, "propulsion/engine[0]/thrust-lbs", np.nan)
        thrust_1_lbs = safe_get(jsb, "propulsion/engine[1]/thrust-lbs", np.nan)
        flap_pos = safe_get(jsb, "fcs/flap-pos-norm", np.nan)
        gear_pos = safe_get(jsb, "gear/gear-pos-norm", np.nan)

        apply_jsbsim_controls(
            jsb,
            elevator_cmd_norm=float(u),
            throttle_cmd=float(throttle)
        )
        jsb.run()

        after = get_jsbsim_state(jsb)

        der = finite_difference(before, after, fallback_dt=dt_local)

        cost_value = (
            (der["dV"] / 0.1) ** 2
            + (der["dgamma"] / 0.001) ** 2
            + (der["dq"] / 0.001) ** 2
        )

        diag = {
            "actual_elevator_rad": actual_elevator_rad,
            "thrust_0_lbs": thrust_0_lbs,
            "thrust_1_lbs": thrust_1_lbs,
            "flap_pos": flap_pos,
            "gear_pos": gear_pos,
        }

        return cost_value, before, der, diag

    def cost(params):
        value, _, _, _ = evaluate(params)
        return value

    bounds = [
        (-5.0, 25.0),  # theta_deg
        (-1.0, 1.0),  # u
        (0.0, 1.0),  # throttle
    ]

    x0 = np.array([theta0_deg, u0, throttle0])

    res = minimize(
        cost,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "ftol": 1e-10,
            "gtol": 1e-10,
            "maxiter": 100,
        }
    )

    theta_deg_opt, u_opt, throttle_opt = res.x

    final_cost, state, der, diag = evaluate(res.x)



    trim_result = {
        "scenario_name": target.name,
        "backend": "jsbsim",
        "config": target.config.name,
        "V": target.V,
        "h": target.h,
        "gamma_deg": target.gamma_deg,
        "theta_deg": float(theta_deg_opt),
        "alpha_deg": float(theta_deg_opt - target.gamma_deg),
        "u": float(u_opt),
        "throttle": float(throttle_opt),
        "dV": float(der["dV"]),
        "dgamma": float(der["dgamma"]),
        "dq": float(der["dq"]),
        "success": bool(res.success),
        "trim_pass": trim_passed(der["dV"], der["dgamma"], der["dq"]),
        "cost": float(final_cost),
        "actual_elevator_rad": float(diag["actual_elevator_rad"]),
        "actual_elevator_deg": float(np.degrees(diag["actual_elevator_rad"])),
        "thrust_0_lbs": float(diag["thrust_0_lbs"]),
        "thrust_1_lbs": float(diag["thrust_1_lbs"]),
        "flap_pos": float(diag["flap_pos"]),
        "gear_pos": float(diag["gear_pos"]),
        "actual_V": float(state["V"]),
        "actual_h": float(state["h"]),
        "actual_theta_deg": float(np.degrees(state["theta"])),
        "actual_gamma_deg": float(np.degrees(state["gamma"])),
        "actual_alpha_deg": float(np.degrees(state["alpha"])),
        "actual_q": float(state["q"]),
    }

    return trim_result


def apply_jsbsim_configuration(jsb, config):
    config_name = getattr(config, "name", "").lower()

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

    return {
        "flap_pos": safe_get(jsb, "fcs/flap-pos-norm", np.nan),
        "gear_pos": safe_get(jsb, "gear/gear-pos-norm", np.nan),
    }


def save_trim_results_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    aircraft = BOEING_737_JSBSIM

    rows = []

    for target in TRIM_TARGETS_737:
        print(f"\n=== Finding trim: {target.name} ===")

        own_trim = find_own_model_trim(
            aircraft=aircraft,
            config=target.config,
            V=target.V,
            h=target.h,
            gamma_deg=target.gamma_deg,
            name=target.name,
        )

        rows.append(own_trim)

        jsb_trim = find_jsbsim_trim(
            target=target,
            theta0_deg=own_trim["theta_deg"],
            u0=own_trim["u"],
            throttle0=own_trim["throttle"],
            dt_local=0.001,
            rotation_mode="6dof",
        )

        rows.append(jsb_trim)

        print("\nOwn trim:")
        print(own_trim)

        print("\nJSBSim trim:")
        print(jsb_trim)

    save_trim_results_csv(
        "results/jsbsim_validation/trim_results_737.csv",
        rows
    )

    print("\nSaved trim results to results/jsbsim_validation/trim_results_737.csv")


if __name__ == "__main__":
    main()