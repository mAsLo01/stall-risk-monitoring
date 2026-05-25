import os
import csv
import sys
from dataclasses import dataclass

import numpy as np
from jsbsim import FGFDMExec

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from AircraftModel import BOEING_737_JSBSIM, CLEAN
from model.dynamics import AircraftDynamics
from config import dt


JSBSIM_DATA_PATH = os.path.join(os.getcwd(), "jsbsim")

FT_TO_M = 0.3048
M_TO_FT = 3.280839895
FPS_TO_MPS = 0.3048
MPS_TO_FPS = 3.280839895


@dataclass
class ReducedOrderFitResult:
    CL0: float
    CL_alpha: float
    CL_delta_e: float

    Cd0: float
    k: float
    CD_delta_e: float

    Cm0: float
    Cm_alpha: float
    Cm_delta_e: float
    Cm_q: float

    mean_abs_CL_error: float
    mean_abs_CD_error: float
    mean_abs_Cm_error: float

    max_abs_CL_error: float
    max_abs_CD_error: float
    max_abs_Cm_error: float


def safe_get(jsb, prop_name, default=None):
    try:
        value = jsb.get_property_value(prop_name)
        if value is None:
            return default
        return value
    except Exception:
        return default


def set_if_exists(jsb, prop_name, value):
    try:
        jsb.set_property_value(prop_name, value)
    except Exception:
        pass


def apply_jsbsim_controls(jsb, elevator_cmd_norm=0.0, throttle_cmd=0.0):
    set_if_exists(jsb, "fcs/elevator-cmd-norm", elevator_cmd_norm)
    set_if_exists(jsb, "fcs/throttle-cmd-norm", throttle_cmd)
    set_if_exists(jsb, "fcs/throttle-cmd-norm[0]", throttle_cmd)
    set_if_exists(jsb, "fcs/throttle-cmd-norm[1]", throttle_cmd)
    set_if_exists(jsb, "propulsion/engine[0]/throttle", throttle_cmd)
    set_if_exists(jsb, "propulsion/engine[1]/throttle", throttle_cmd)


def get_jsbsim_state(jsb):
    V = safe_get(jsb, "velocities/vt-fps", 0.0) * FPS_TO_MPS

    theta = safe_get(jsb, "attitude/theta-rad", None)
    if theta is None:
        theta = safe_get(jsb, "attitude/pitch-rad", 0.0)

    q = safe_get(jsb, "velocities/q-rad_sec", 0.0)

    h = safe_get(jsb, "position/h-sl-ft", 0.0) * FT_TO_M

    alpha = safe_get(jsb, "aero/alpha-rad", None)
    if alpha is None:
        alpha = np.radians(safe_get(jsb, "aero/alpha-deg", 0.0))

    gamma = safe_get(jsb, "flight-path/gamma-rad", None)
    if gamma is None:
        gamma = safe_get(jsb, "velocities/gamma-rad", None)

    if gamma is None:
        hdot_fps = safe_get(jsb, "velocities/h-dot-fps", 0.0)
        hdot = hdot_fps * FPS_TO_MPS
        gamma = np.arcsin(np.clip(hdot / max(V, 1e-9), -1.0, 1.0))

    t = safe_get(jsb, "simulation/sim-time-sec", 0.0)

    return {
        "t": t,
        "V": V,
        "theta": theta,
        "gamma": gamma,
        "alpha": alpha,
        "q": q,
        "h": h,
    }


def finite_difference(before, after, fallback_dt):
    dt_actual = after["t"] - before["t"]

    if not np.isfinite(dt_actual) or dt_actual <= 0.0:
        dt_actual = fallback_dt

    return {
        "dt": dt_actual,
        "dV": (after["V"] - before["V"]) / dt_actual,
        "dtheta": (after["theta"] - before["theta"]) / dt_actual,
        "dgamma": (after["gamma"] - before["gamma"]) / dt_actual,
        "dh": (after["h"] - before["h"]) / dt_actual,
        "dq": (after["q"] - before["q"]) / dt_actual,
    }


def compute_required_coefficients(
        before,
        derivatives,
        aircraft,
        dynamics,
        delta_rad,
        throttle_cmd=0.0,
):
    """
    Восстанавливает CL, CD, Cm, которые требуются reduced-order модели,
    чтобы дать такие же dV, dgamma, dq, как JSBSim в данной точке.
    """

    V = before["V"]
    gamma = before["gamma"]
    alpha = before["alpha"]
    h = before["h"]
    q = before["q"]

    rho = dynamics.atmosphere.density(h)
    q_dyn = 0.5 * rho * V ** 2

    S = aircraft.wing_area
    c = aircraft.mean_chord
    m = aircraft.mass
    g = dynamics.g
    Iyy = aircraft.Iyy

    # Для static identification лучше держать throttle=0.
    # Но если throttle не ноль, используем собственную модель тяги как приближение.
    T = dynamics.compute_thrust(throttle_cmd, V, alpha)
    T_x = T * np.cos(alpha)
    T_z = T * np.sin(alpha)

    dV_jsb = derivatives["dV"]
    dgamma_jsb = derivatives["dgamma"]
    dq_jsb = derivatives["dq"]

    # dV = (T_x - D)/m - g*sin(gamma)
    D_required = T_x - m * (dV_jsb + g * np.sin(gamma))
    CD_required = D_required / max(q_dyn * S, 1e-9)

    # dgamma = (L + T_z - m*g*cos(gamma))/(m*V)
    L_required = dgamma_jsb * m * V + m * g * np.cos(gamma) - T_z
    CL_required = L_required / max(q_dyn * S, 1e-9)

    # dq = M/Iyy, M = q_dyn*S*c*Cm
    Cm_required = dq_jsb * Iyy / max(q_dyn * S * c, 1e-9)

    q_hat = q * c / max(2.0 * V, 1e-9)

    return {
        "rho": rho,
        "q_dyn": q_dyn,

        "alpha_rad": alpha,
        "alpha_deg": np.degrees(alpha),

        "gamma_rad": gamma,
        "gamma_deg": np.degrees(gamma),

        "theta_rad": before["theta"],
        "theta_deg": np.degrees(before["theta"]),

        "q": q,
        "q_hat": q_hat,

        "V": V,
        "h": h,

        "delta_rad": delta_rad,
        "delta_deg": np.degrees(delta_rad),

        "CL_required": CL_required,
        "CD_required": CD_required,
        "Cm_required": Cm_required,

        "L_required": L_required,
        "D_required": D_required,

        "dV_jsb": dV_jsb,
        "dgamma_jsb": dgamma_jsb,
        "dq_jsb": dq_jsb,
    }


def make_jsbsim_instance(
        V_mps,
        h_m,
        alpha_deg,
        gamma_deg,
        dt_local,
        rotation_mode="6dof",
        q_rad_s=0.0,
):
    jsb = FGFDMExec(JSBSIM_DATA_PATH)
    jsb.load_model("737")
    jsb.set_dt(dt_local)

    theta_deg = gamma_deg + alpha_deg

    jsb.set_property_value("ic/h-sl-ft", h_m * M_TO_FT)
    jsb.set_property_value("ic/vt-fps", V_mps * MPS_TO_FPS)
    jsb.set_property_value("ic/alpha-deg", alpha_deg)
    jsb.set_property_value("ic/theta-deg", theta_deg)
    jsb.set_property_value("ic/gamma-deg", gamma_deg)

    jsb.set_property_value("ic/phi-deg", 0.0)
    jsb.set_property_value("ic/psi-deg", 0.0)
    jsb.set_property_value("ic/lat-gc-deg", 0.0)
    jsb.set_property_value("ic/long-gc-deg", 0.0)

    jsb.set_property_value("ic/q-rad_sec", q_rad_s)
    jsb.set_property_value("ic/p-rad_sec", 0.0)
    jsb.set_property_value("ic/r-rad_sec", 0.0)

    if rotation_mode == "3dof":
        jsb.set_property_value("simulation/rotation", 0)
    else:
        jsb.set_property_value("simulation/rotation", 1)

    jsb.run_ic()

    return jsb


def collect_static_aero_samples_from_jsbsim(
        alpha_grid_deg=None,
        elevator_grid_deg=None,
        q_grid_rad_s=None,
        V_grid_mps=None,
        h_m=500.0,
        gamma_deg=-3.92,
        dt_local=0.001,
        rotation_mode="6dof",
        elevator_max_rad=0.30,
        throttle_cmd=0.0,
):
    """
    Собирает JSBSim-сэмплы и для каждой точки восстанавливает:
    CL_required, CD_required, Cm_required.

    Это замена отдельных функций для D, L и Cm.
    """

    if alpha_grid_deg is None:
        alpha_grid_deg = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    if elevator_grid_deg is None:
        elevator_grid_deg = [-10.0, -5.0, 0.0, 5.0, 10.0]

    if V_grid_mps is None:
        V_grid_mps = [70.0, 80.0, 90.0]

    if q_grid_rad_s is None:
        q_grid_rad_s = [-0.08, -0.04, 0.0, 0.04, 0.08]

    aircraft = BOEING_737_JSBSIM
    dynamics = AircraftDynamics(aircraft, CLEAN)

    rows = []

    print("\n=== Collecting static aero samples from JSBSim ===")
    print(
        f"{'V':>8} | "
        f"{'alpha':>8} | "
        f"{'delta':>8} | "
        f"{'CL_req':>10} | "
        f"{'CD_req':>10} | "
        f"{'Cm_req':>10}"
    )
    print("-" * 70)

    for V_mps in V_grid_mps:
        for alpha_deg in alpha_grid_deg:
            for elevator_deg in elevator_grid_deg:
                for q_cmd in q_grid_rad_s:
                    desired_delta_rad = np.radians(elevator_deg)

                    u_cmd = desired_delta_rad / elevator_max_rad
                    u_cmd = float(np.clip(u_cmd, -1.0, 1.0))

                    jsb = make_jsbsim_instance(
                        V_mps=V_mps,
                        h_m=h_m,
                        alpha_deg=alpha_deg,
                        gamma_deg=gamma_deg,
                        dt_local=dt_local,
                        rotation_mode=rotation_mode,
                        q_rad_s=q_cmd,
                    )

                    apply_jsbsim_controls(
                        jsb,
                        elevator_cmd_norm=u_cmd,
                        throttle_cmd=throttle_cmd,
                    )

                    # Первый шаг — чтобы FCS применил команду.
                    jsb.run()

                    before = get_jsbsim_state(jsb)

                    delta_rad = safe_get(jsb, "fcs/elevator-pos-rad", None)
                    if delta_rad is None:
                        delta_rad = safe_get(jsb, "fcs/elevator-control", None)

                    if delta_rad is None:
                        raise RuntimeError("Не удалось прочитать elevator-pos-rad/control из JSBSim.")

                    # Второй шаг — производные.
                    jsb.run()
                    after = get_jsbsim_state(jsb)

                    der = finite_difference(before, after, fallback_dt=dt_local)

                    row = compute_required_coefficients(
                        before=before,
                        derivatives=der,
                        aircraft=aircraft,
                        dynamics=dynamics,
                        delta_rad=delta_rad,
                        throttle_cmd=throttle_cmd,
                    )

                    row["V_cmd"] = V_mps
                    row["alpha_cmd_deg"] = alpha_deg
                    row["elevator_cmd_deg"] = elevator_deg
                    row["u_cmd"] = u_cmd
                    row["dt_actual"] = der["dt"]
                    row["q_cmd"] = q_cmd

                    rows.append(row)

                    print(
                        f"{'V':>8} | "
                        f"{'alpha':>8} | "
                        f"{'delta':>8} | "
                        f"{'q':>8} | "
                        f"{'q_hat':>10} | "
                        f"{'CL_req':>10} | "
                        f"{'CD_req':>10} | "
                        f"{'Cm_req':>10}"
                    )
                    print("-" * 92)
                    print(
                        f"{V_mps:8.2f} | "
                        f"{row['alpha_deg']:8.3f} | "
                        f"{row['delta_deg']:8.3f} | "
                        f"{row['q']:8.4f} | "
                        f"{row['q_hat']:10.6f} | "
                        f"{row['CL_required']:10.5f} | "
                        f"{row['CD_required']:10.5f} | "
                        f"{row['Cm_required']:10.5f}"
                    )

    return rows


def fit_reduced_order_coefficients(
        samples,
        fit_k=True,
        fit_cl_delta_e=True,
        fit_cd_delta_e=True,
        fit_cm_q=True,
):
    """
    Одновременно по одному набору JSBSim-сэмплов подгоняет reduced-order модель:

        CL = CL0 + CL_alpha * alpha + CL_delta_e * delta

        CD = Cd0 + k * CL^2 + CD_delta_e * delta^2

        Cm = Cm0 + Cm_alpha * alpha + Cm_delta_e * delta + Cm_q * q_hat

    alpha, delta в радианах.
    q_hat = q * c / (2V)
    """

    aircraft = BOEING_737_JSBSIM

    alpha = np.array([s["alpha_rad"] for s in samples], dtype=float)
    delta = np.array([s["delta_rad"] for s in samples], dtype=float)
    q_hat = np.array([s["q_hat"] for s in samples], dtype=float)

    CL_req = np.array([s["CL_required"] for s in samples], dtype=float)
    CD_req = np.array([s["CD_required"] for s in samples], dtype=float)
    Cm_req = np.array([s["Cm_required"] for s in samples], dtype=float)

    # =========================
    # 1. Lift model
    # =========================
    if fit_cl_delta_e:
        X_cl = np.column_stack([
            np.ones_like(alpha),
            alpha,
            delta,
        ])

        cl_coef, *_ = np.linalg.lstsq(X_cl, CL_req, rcond=None)
        CL0, CL_alpha, CL_delta_e = cl_coef
    else:
        X_cl = np.column_stack([
            np.ones_like(alpha),
            alpha,
        ])

        cl_coef, *_ = np.linalg.lstsq(X_cl, CL_req, rcond=None)
        CL0, CL_alpha = cl_coef
        CL_delta_e = 0.0

    CL_pred = CL0 + CL_alpha * alpha + CL_delta_e * delta

    # =========================
    # 2. Drag model
    # =========================
    if fit_k and fit_cd_delta_e:
        X_cd = np.column_stack([
            np.ones_like(CL_pred),
            CL_pred ** 2,
            delta ** 2,
        ])

        cd_coef, *_ = np.linalg.lstsq(X_cd, CD_req, rcond=None)
        Cd0, k, CD_delta_e = cd_coef

    elif fit_k and not fit_cd_delta_e:
        X_cd = np.column_stack([
            np.ones_like(CL_pred),
            CL_pred ** 2,
        ])

        cd_coef, *_ = np.linalg.lstsq(X_cd, CD_req, rcond=None)
        Cd0, k = cd_coef
        CD_delta_e = 0.0

    elif not fit_k and fit_cd_delta_e:
        k = aircraft.k

        X_cd = np.column_stack([
            np.ones_like(CL_pred),
            delta ** 2,
        ])

        y_cd = CD_req - k * CL_pred ** 2
        cd_coef, *_ = np.linalg.lstsq(X_cd, y_cd, rcond=None)
        Cd0, CD_delta_e = cd_coef

    else:
        k = aircraft.k
        CD_delta_e = 0.0
        Cd0 = float(np.mean(CD_req - k * CL_pred ** 2))

    CD_pred = Cd0 + k * CL_pred ** 2 + CD_delta_e * delta ** 2

    # =========================
    # 3. Pitch moment model
    # =========================
    if fit_cm_q:
        X_cm = np.column_stack([
            np.ones_like(alpha),
            alpha,
            delta,
            q_hat,
        ])

        cm_coef, *_ = np.linalg.lstsq(X_cm, Cm_req, rcond=None)
        Cm0, Cm_alpha, Cm_delta_e, Cm_q = cm_coef

        Cm_pred = (
            Cm0
            + Cm_alpha * alpha
            + Cm_delta_e * delta
            + Cm_q * q_hat
        )
    else:
        X_cm = np.column_stack([
            np.ones_like(alpha),
            alpha,
            delta,
        ])

        cm_coef, *_ = np.linalg.lstsq(X_cm, Cm_req, rcond=None)
        Cm0, Cm_alpha, Cm_delta_e = cm_coef
        Cm_q = aircraft.Cm_q

        Cm_pred = (
            Cm0
            + Cm_alpha * alpha
            + Cm_delta_e * delta
        )

    result = ReducedOrderFitResult(
        CL0=float(CL0),
        CL_alpha=float(CL_alpha),
        CL_delta_e=float(CL_delta_e),

        Cd0=float(Cd0),
        k=float(k),
        CD_delta_e=float(CD_delta_e),

        Cm0=float(Cm0),
        Cm_alpha=float(Cm_alpha),
        Cm_delta_e=float(Cm_delta_e),
        Cm_q=float(Cm_q),

        mean_abs_CL_error=float(np.mean(np.abs(CL_req - CL_pred))),
        mean_abs_CD_error=float(np.mean(np.abs(CD_req - CD_pred))),
        mean_abs_Cm_error=float(np.mean(np.abs(Cm_req - Cm_pred))),

        max_abs_CL_error=float(np.max(np.abs(CL_req - CL_pred))),
        max_abs_CD_error=float(np.max(np.abs(CD_req - CD_pred))),
        max_abs_Cm_error=float(np.max(np.abs(Cm_req - Cm_pred))),
    )

    print("\n=== Simultaneous reduced-order coefficient fit ===")
    print(f"CL0           = {result.CL0:+.6f}")
    print(f"CL_alpha      = {result.CL_alpha:+.6f} 1/rad")
    print(f"CL_delta_e    = {result.CL_delta_e:+.6f} 1/rad")
    print(f"Cd0           = {result.Cd0:+.6f}")
    print(f"k             = {result.k:+.6f}")
    print(f"CD_delta_e    = {result.CD_delta_e:+.6f} 1/rad^2")
    print(f"Cm0           = {result.Cm0:+.6f}")
    print(f"Cm_alpha      = {result.Cm_alpha:+.6f} 1/rad")
    print(f"Cm_delta_e    = {result.Cm_delta_e:+.6f} 1/rad")
    print(f"Cm_q          = {result.Cm_q:+.6f} 1/rad")

    print("\n=== Fit errors ===")
    print(f"mean |CL error| = {result.mean_abs_CL_error:.6f}")
    print(f"mean |CD error| = {result.mean_abs_CD_error:.6f}")
    print(f"mean |Cm error| = {result.mean_abs_Cm_error:.6f}")
    print(f"max  |CL error| = {result.max_abs_CL_error:.6f}")
    print(f"max  |CD error| = {result.max_abs_CD_error:.6f}")
    print(f"max  |Cm error| = {result.max_abs_Cm_error:.6f}")

    print("\n=== Recommended AircraftModel update ===")
    print(f"BOEING_737_JSBSIM.CL0 = {result.CL0:+.6f}")
    print(f"BOEING_737_JSBSIM.CL_alpha_override = {result.CL_alpha:+.6f}")
    print(f"BOEING_737_JSBSIM.CL_delta_e = {result.CL_delta_e:+.6f}")
    print(f"BOEING_737_JSBSIM.Cd0 = {result.Cd0:+.6f}")
    print(f"BOEING_737_JSBSIM.k_override = {result.k:+.6f}")
    print(f"BOEING_737_JSBSIM.CD_delta_e = {result.CD_delta_e:+.6f}")
    print(f"BOEING_737_JSBSIM.Cm0 = {result.Cm0:+.6f}")
    print(f"BOEING_737_JSBSIM.Cm_alpha = {result.Cm_alpha:+.6f}")
    print(f"BOEING_737_JSBSIM.Cm_delta_e = {result.Cm_delta_e:+.6f}")
    print(f"BOEING_737_JSBSIM.Cm_q = {result.Cm_q:+.6f}")

    return result



def save_fit_result_csv(path, result: ReducedOrderFitResult):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = {
        "CL0": result.CL0,
        "CL_alpha": result.CL_alpha,
        "CL_delta_e": result.CL_delta_e,

        "Cd0": result.Cd0,
        "k": result.k,
        "CD_delta_e": result.CD_delta_e,

        "Cm0": result.Cm0,
        "Cm_alpha": result.Cm_alpha,
        "Cm_delta_e": result.Cm_delta_e,
        "Cm_q": result.Cm_q,

        "mean_abs_CL_error": result.mean_abs_CL_error,
        "mean_abs_CD_error": result.mean_abs_CD_error,
        "mean_abs_Cm_error": result.mean_abs_Cm_error,

        "max_abs_CL_error": result.max_abs_CL_error,
        "max_abs_CD_error": result.max_abs_CD_error,
        "max_abs_Cm_error": result.max_abs_Cm_error,
    }

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_samples_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    samples = collect_static_aero_samples_from_jsbsim(
        alpha_grid_deg=[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        elevator_grid_deg=[-10.0, -5.0, 0.0, 5.0, 10.0],
        q_grid_rad_s=[-0.08, -0.04, 0.0, 0.04, 0.08],
        V_grid_mps=[70.0, 80.0, 90.0],
        h_m=500.0,
        gamma_deg=-3.92,
        dt_local=0.001,
        rotation_mode="6dof",
        elevator_max_rad=0.30,
        throttle_cmd=0.0,
    )

    save_samples_csv(
        "results/jsbsim_validation/static_aero_pitchrate_samples.csv",
        samples,
    )

    result = fit_reduced_order_coefficients(
        samples,
        fit_k=True,
        fit_cl_delta_e=True,
        fit_cd_delta_e=True,
        fit_cm_q=True,
    )

    save_fit_result_csv(
        "results/jsbsim_validation/static_fit_result_with_cmq.csv",
        result,
    )


if __name__ == "__main__":
    main()