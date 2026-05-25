from dataclasses import dataclass
import numpy as np



def kcas_to_mps(kcas):
    return kcas * 0.514444


def compute_CLmax_from_stall_speed(mass, wing_area, Vs_mps, rho=1.225, g=9.81):
    W = mass * g
    return 2.0 * W / (rho * wing_area * Vs_mps ** 2)


def finalize_config(config, aircraft, rho=1.225):
    Vs = kcas_to_mps(config.stall_speed_kcas)
    config.CL_max = compute_CLmax_from_stall_speed(
        aircraft.mass,
        aircraft.wing_area,
        Vs,
        rho=rho
    )
    return config
@dataclass
class AeroConfig:
    name: str
    flap_deg: float
    gear_down: bool
    stall_speed_kcas: float
    dCL0: float
    dCd0: float
    dCm0: float
    dAlpha_crit: float

    CL_max: float = 0.0

    # поправка эффективности руля высоты по конфигурации
    elevator_moment_gain: float = 1.0
# BOEING_737
CLEAN = AeroConfig(
    name="clean",
    flap_deg=0.0,
    gear_down=False,
    stall_speed_kcas=128.0,
    dCL0=0.0,
    dCd0=0.0,
    dCm0=0.0,
    dAlpha_crit=0.0,
    elevator_moment_gain=1.0,
)
LANDING = AeroConfig(
    name="landing",
    flap_deg=30.0,
    gear_down=True,
    stall_speed_kcas=118.0,
    dCL0=0.60,
    dCd0=0.010,
    dCm0=0.01,
    dAlpha_crit=np.radians(-2.0),
    elevator_moment_gain=0.90,
)

# CESSNA 172S
# # Закрылки убраны, шасси убраны, поправок к аэродинамике нет: минимальное сопротивление, минимальный CL_max, наибольшая скорость сваливания
# CLEAN = AeroConfig(
#     name="clean",
#     flap_deg=0.0,
#     gear_down=False,
#     stall_speed_kcas=53.0,
#     dCL0=0.0,
#     dCd0=0.0,
#     dCm0=0.0,
#     dAlpha_crit=0.0,
# )
#
#
# # посадочная: закрылки выпущены (30 град), шасси выпущено, максимальная подъемная конфигурация
# # сильно увеличивается коэффициент подъемной силы, сильно увеличивается сопротивление, скорость сваливания уменьшается
# LANDING = AeroConfig(
#     name="landing",
#     flap_deg=30.0,
#     gear_down=True,
#     stall_speed_kcas=48.0,
#     dCL0=0.35,
#     dCd0=0.055,
#     dCm0=-0.08,
#     dAlpha_crit=np.radians(-2.5),
# )

@dataclass
class AircraftModel:
    name: str

    # базовая геометрия и масса
    mass: float
    wing_area: float
    wing_span: float
    mean_chord: float | None

    Iyy: float

    # аэродинамика: базовые источниковые/выводимые параметры
    CL0: float
    CL_alpha_override: float | None
    CL_delta_e: float
    k_override: float | None
    stall_speed_clean_kcas: float
    alpha_stall_on: float
    alpha_stall_off: float

    oswald_e: float
    Cd0: float
    CD_delta_e: float

    # моменты — калибруемые/из открытых моделей
    Cm0: float
    Cm_alpha: float
    Cm_q: float
    Cm_delta_e: float

    # post-stall
    CL_plateau_ratio: float
    CL_vortex_gain: float

    tau_sep_stall: float
    tau_sep_recover: float

    max_elevator: float
    engine_power: float
    prop_efficiency: float
    static_thrust_max: float

    def __post_init__(self):
        self.AR = self.wing_span ** 2 / self.wing_area

        if self.mean_chord is None:
            self.mean_chord = self.wing_area / self.wing_span

        if self.k_override is not None:
            self.k = self.k_override
        else:
            self.k = 1.0 / (np.pi * self.AR * self.oswald_e)

        # согласуем CL_alpha с CL0, CLmax и alpha_stall
        Vs_clean = kcas_to_mps(self.stall_speed_clean_kcas)

        self.CL_max_clean = compute_CLmax_from_stall_speed(
            self.mass,
            self.wing_area,
            Vs_clean
        )

        if self.CL_alpha_override is not None:
            self.CL_alpha = self.CL_alpha_override
        else:
            self.CL_alpha = (self.CL_max_clean - self.CL0) / self.alpha_stall_on

CESSNA_172 = AircraftModel(
    name="Cessna 172S",

    mass=2550 * 0.45359237,
    wing_area=16.17,
    wing_span=11.00,
    mean_chord=1.49,

    Iyy=1825.0,

    CL0=0.31,
    CL_alpha_override = None,
    CL_delta_e=2.0,
    stall_speed_clean_kcas=53.0,
    alpha_stall_on=np.radians(15.0),
    alpha_stall_off=np.radians(11.5),

    oswald_e=0.80,
    Cd0=0.031,
    CD_delta_e=0.0,
    k_override = None,

    Cm0=-0.015,
    Cm_alpha=-0.89,
    Cm_q=-12.4,

    Cm_delta_e=1.28,

    CL_plateau_ratio=0.45,
    CL_vortex_gain=0.10,

    tau_sep_stall=0.25,
    tau_sep_recover=0.8,

    max_elevator=np.radians(25.0),
    engine_power=180.0 * 745.7,  # 180 hp в ваттах
    prop_efficiency=0.78,
    static_thrust_max=2200.0,

)
# CLEAN = finalize_config(CLEAN, CESSNA_172)
# LANDING = finalize_config(LANDING, CESSNA_172)

BOEING_737_800 = AircraftModel(
    name="Boeing 737-800",
    mass=73500,                # кг
    wing_area=125,             # м²
    wing_span=35,              # м
    mean_chord=3.75,           # м
    Iyy=3_660_000,             # кг·м²
    CL0=0.35,
    CL_delta_e=0.30,
    CL_alpha_override = None,
    stall_speed_clean_kcas=128,
    alpha_stall_on=np.radians(16.0),
    alpha_stall_off=np.radians(11.5),
    oswald_e=0.86,
    Cd0=0.015,
    CD_delta_e=0.0,
    k_override = None,
    Cm0=-0.05,
    Cm_alpha=-0.8,
    Cm_q=-15.0,
    Cm_delta_e=-0.9,
    CL_plateau_ratio=0.50,
    CL_vortex_gain=0.12,
    tau_sep_stall=0.30,
    tau_sep_recover=1.0,
    max_elevator=np.radians(20.0),
    engine_power=2 * 107_600,  # суммарная статическая тяга двух двигателей, Н
    prop_efficiency=0.85,
    static_thrust_max=2 * 107_600  # Н
)
# CLEAN = finalize_config(CLEAN, BOEING_737_800)
# LANDING = finalize_config(LANDING, BOEING_737_800)

BOEING_737_JSBSIM = AircraftModel(
    name="Boeing 737 (JSBSim)",
    mass=48534,                              # кг (107 000 lb с топливом)
    wing_area=108.8,  # 91.04 проверить, найдено в открытых источниках                       # м² (1171 ft²)
    wing_span=28.86,                         # м (94.7 ft)
    mean_chord=3.75, # =  wing_area/wing_span~3.15 при 91.04                     # м (12.31 ft)
    Iyy=2_087_100,                           # кг·м² (1.473×10⁶ slug·ft²)
    CL0=0.205685,                                # из Lift_due_to_alpha при α=0
    CL_delta_e=0.303153,
    CL_alpha_override=4.389258,
    stall_speed_clean_kcas=128,              # оценка (для 737‑300)
    alpha_stall_on=np.radians(16.0),         # оценка
    alpha_stall_off=np.radians(11.5),        # оценка
    oswald_e=0.97,                           # из индуктивного сопротивления (k=0.043)
    Cd0=0.049891,                               # из таблицы Drag_due_to_alpha при α=0
    CD_delta_e=0.434094,
    k_override=0.055481,
    Cm0=-0.040675,
    Cm_alpha=-1.083733,
    Cm_delta_e=-1.130547,
    Cm_q=-43.517326,                              # из Pitch_moment_due_to_pitch_rate
    CL_plateau_ratio=0.50,                   # оценка
    CL_vortex_gain=0.12,                     # оценка
    tau_sep_stall=0.30,                      # оценка
    tau_sep_recover=1.0,                     # оценка
    max_elevator=np.radians(17.2),           # ограничение в FCS (±0.3 рад)
    engine_power=2 * 89_000,                 # тяга двух CFM56-3 (≈89 кН каждый)
    prop_efficiency=0.85,                    # оценка
    static_thrust_max=2 * 89_000             # Н
)
CLEAN = finalize_config(CLEAN, BOEING_737_JSBSIM)
LANDING = finalize_config(LANDING, BOEING_737_JSBSIM)