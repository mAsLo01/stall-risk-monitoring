from dataclasses import dataclass

@dataclass
class State:
    theta: float = 0.1
    q: float = 0.0
    V: float = 50.0
    gamma: float = 0.08
    h: float = 300.0
    x: float = 0.0

    R: float = 0.0
    alpha: float = 0.0
    alpha_prev: float = 0.0
    alpha_dot: float = 0.0

    sep: float = 0.0

    CL: float = 0.0
    CD: float = 0.0
    Cm: float = 0.0
    CL_max: float = 0.0
    CL_ratio: float = 0.0

    elevator_eff: float = 1.0

    V_stall_1g: float = 0.0
    V_stall_maneuver: float = 0.0
    speed_margin_1g: float = 1.0
    speed_margin_maneuver: float = 1.0

    # Совместимость
    V_stall: float = 0.0
    speed_margin: float = 1.0

    dCL_turb: float = 0.0
    dCD_turb: float = 0.0
    dCm_turb: float = 0.0

    throttle: float = 0.0
    thrust: float = 0.0

    vertical_speed: float = 0.0
    power_available: float = 0.0
    power_required_total: float = 0.0
    excess_power: float = 0.0
    energy_margin: float = 0.0

    load_factor: float = 1.0

    stall_warning: bool = False
    stall_warning_speed: float = 0.0
    stall_warning_margin: float = 0.0

    sep_target: float = 0.0
    pitch_damping_eff: float = 1.0
    elevator_delta: float = 0.0
    elevator_command: float = 0.0

    run_id = None
    seed = None
    parameter_name = None
    parameter_variant = None
    parameter_value = None

    mode: str = "NORMAL"