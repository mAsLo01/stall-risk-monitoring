# experiments/scenarios_737.py

from dataclasses import dataclass
from typing import Callable
import numpy as np
from config import dt
from AircraftModel import CLEAN, LANDING
from model.state import State


ControlLaw = Callable[[float, State], tuple[float, float]]


@dataclass
class Scenario:
    name: str
    description: str
    config: object
    t_final: float
    make_initial_state: Callable[[], State]
    control_law: ControlLaw

    # Новые поля — не ломают старый код, потому что имеют значения по умолчанию
    group: str = "unspecified"
    purpose: str = ""
    parent_trim: str | None = None
    primary_for_validation: bool = False


def pitch_hold_controller_trim(theta_cmd_deg, throttle_cmd, u_trim, Kp=8.0, Kd=4.0):
    theta_cmd_rad = np.radians(theta_cmd_deg)
    def control(t, state):
        u = -Kp * (theta_cmd_rad - state.theta) + Kd * state.q + u_trim
        return np.clip(u, -1, 1), throttle_cmd
    return control

def elevator_jam_controller(
    gamma_cmd_deg, throttle_trim, V_cmd, u_trim,
    t_jam, jam_elevator_cmd=-0.5, jam_effect_mult=0.1
):
    """
    До t_jam: нормальный gamma_speed_controller.
    После t_jam: фиксированный руль (кабрирование) и резкое падение эффективности.
    """
    gamma_ctrl = gamma_speed_controller(gamma_cmd_deg, throttle_trim, V_cmd, u_trim)
    jammed = False

    def control(t, state):
        nonlocal jammed
        if t < t_jam:
            # восстанавливаем множитель, если был изменён
            from model import dynamics
            dynamics.ELEVATOR_EFFECT_MULTIPLIER = 1.0
            return gamma_ctrl(t, state)
        else:
            if not jammed:
                from model import dynamics
                dynamics.ELEVATOR_EFFECT_MULTIPLIER = jam_effect_mult
                jammed = True
            # фиксированное отклонение руля (кабрирование) и крейсерская тяга
            return np.clip(jam_elevator_cmd, -1, 1), throttle_trim
    return control

def tk1951_controller(
    gamma_cmd_deg, throttle_trim, V_cmd, u_trim,
    t_fail, fail_throttle,
    pitch_schedule,                # до начала ухода
    go_around_time,                # момент начала ухода
    go_around_throttle,            # взлётная тяга
    go_around_pitch_cmd_deg,       # команда на опускание носа
    Kp_pitch=8.0, Kd_pitch=4.0
):
    """
    Фазы:
    0..t_fail: нормальный заход (gamma_speed_controller).
    t_fail..go_around_time: отказ, пилот тянет нос на себя (pitch_schedule).
    > go_around_time: уход на второй круг – взлётная тяга и команда на пикирование.
    """
    gamma_ctrl = gamma_speed_controller(gamma_cmd_deg, throttle_trim, V_cmd, u_trim)
    fail_ctrl = scheduled_trim_controller(
        schedule=pitch_schedule, u_trim=u_trim, Kp=Kp_pitch, Kd=Kd_pitch
    )
    go_around_ctrl = pitch_hold_controller_trim(
        theta_cmd_deg=go_around_pitch_cmd_deg,
        throttle_cmd=go_around_throttle,
        u_trim=u_trim,
        Kp=Kp_pitch, Kd=Kd_pitch
    )
    failed = False
    go_around = False

    def control(t, state):
        nonlocal failed, go_around
        if t < t_fail:
            return gamma_ctrl(t, state)
        elif t < go_around_time:
            if not failed:
                gamma_ctrl.reset_integrals()   # если у gamma_speed_controller есть метод reset
                failed = True
            return fail_ctrl(t, state)
        else:
            if not go_around:
                go_around = True
            return go_around_ctrl(t, state)
    return control

def af447_controller(
    gamma_cmd_deg, throttle_trim, V_cmd, u_trim,
    t_fail, cruise_throttle, pitch_cmd_deg,
    Kp_pitch=8.0, Kd_pitch=4.0
):
    """
    До t_fail: нормальный крейсерский полёт (gamma_speed_controller).
    После t_fail: ручное управление – фиксированный тангаж и тяга.
    """
    gamma_ctrl = gamma_speed_controller(gamma_cmd_deg, throttle_trim, V_cmd, u_trim)
    fail_ctrl = pitch_hold_controller_trim(
        theta_cmd_deg=pitch_cmd_deg,
        throttle_cmd=cruise_throttle,
        u_trim=u_trim,
        Kp=Kp_pitch, Kd=Kd_pitch
    )
    failed = False

    def control(t, state):
        nonlocal failed
        if t < t_fail:
            return gamma_ctrl(t, state)
        else:
            if not failed:
                gamma_ctrl.reset_integrals()
                failed = True
            return fail_ctrl(t, state)
    return control


def gamma_speed_controller(gamma_cmd_deg, throttle_trim, V_cmd, u_trim,
                           Kp_gamma=8.0, Ki_gamma=0.3, Kd_gamma=2.0,
                           Kp_V=0.005, Ki_V=0.0005):
    gamma_cmd_rad = np.radians(gamma_cmd_deg)
    integral_gamma = [0.0]   # список, чтобы можно было менять снаружи
    integral_V = [0.0]
    prev_t = [None]

    def control(t, state):
        if prev_t[0] is None:
            dt_ctrl = 0.01
        else:
            dt_ctrl = t - prev_t[0]
        prev_t[0] = t

        error_gamma = state.gamma - gamma_cmd_rad
        integral_gamma[0] += error_gamma * dt_ctrl
        integral_gamma[0] = np.clip(integral_gamma[0], -0.5, 0.5)

        u = (u_trim
             + Kp_gamma * error_gamma
             + Ki_gamma * integral_gamma[0]
             + Kd_gamma * state.q)

        error_V = V_cmd - state.V
        integral_V[0] += error_V * dt_ctrl
        integral_V[0] = np.clip(integral_V[0], -50.0, 50.0)
        throttle = throttle_trim + Kp_V * error_V + Ki_V * integral_V[0]

        return np.clip(u, -1, 1), np.clip(throttle, 0.0, 1.0)

    # Возможность сброса
    def reset_integrals():
        integral_gamma[0] = 0.0
        integral_V[0] = 0.0
    control.reset_integrals = reset_integrals

    return control

# Cessna_172S
# def pitch_hold_controller(theta_cmd_deg, throttle_cmd, Kp=8, Kd=4):
#     theta_cmd_rad = np.radians(theta_cmd_deg)
#     def control(t, state):
#         u = Kp * (theta_cmd_rad - state.theta) - Kd * state.q
#         return clamp_control(u), throttle_cmd
#     return control

def gamma_speed_with_failure_controller(
    gamma_cmd_deg, throttle_trim, V_cmd, u_trim,
    t_fail, fail_throttle, pitch_schedule,
    Kp_gamma=8.0, Ki_gamma=0.3, Kd_gamma=2.0,
    Kp_V=0.005, Ki_V=0.0005,
    Kp_pitch=8.0, Kd_pitch=4.0
):
    gamma_ctrl = gamma_speed_controller(
        gamma_cmd_deg, throttle_trim, V_cmd, u_trim,
        Kp_gamma, Ki_gamma, Kd_gamma, Kp_V, Ki_V
    )
    pitch_ctrl = scheduled_trim_controller(
        schedule=pitch_schedule, u_trim=u_trim,
        Kp=Kp_pitch, Kd=Kd_pitch
    )
    failed = False

    def control(t, state):
        nonlocal failed
        if t < t_fail:
            return gamma_ctrl(t, state)
        else:
            if not failed:
                gamma_ctrl.reset_integrals()   # сбрасываем интеграторы
                failed = True
            return pitch_ctrl(t, state)

    return control

def scheduled_trim_controller(schedule, u_trim, Kp=8.0, Kd=4.0):
    """
    schedule: список кортежей (t_start, theta_cmd_deg, throttle_cmd)
    Возвращает u и throttle.
    u = Kp*(theta_cmd_rad - theta) - Kd*q + u_trim
    """
    schedule = sorted(schedule, key=lambda x: x[0])

    def control(t, state):
        # Находим актуальную команду
        theta_cmd_deg = schedule[0][1]
        throttle_cmd = schedule[0][2]
        for t_start, theta_deg, thr in schedule:
            if t >= t_start:
                theta_cmd_deg = theta_deg
                throttle_cmd = thr
            else:
                break

        theta_cmd_rad = np.radians(theta_cmd_deg)
        u = u_trim - Kp * (theta_cmd_rad - state.theta) + Kd * state.q
        return np.clip(u, -1, 1), np.clip(throttle_cmd, 0.0, 1.0)

    return control


def base_state(V=80.0, h=1500.0, theta_deg=5.0, gamma_deg=2.0):
    def make():
        state = State()
        state.V = V
        state.h = h
        state.theta = np.radians(theta_deg)
        state.gamma = np.radians(gamma_deg)
        state.q = 0.0
        state.x = 0.0
        state.R = 0.0
        state.sep = 0.0
        return state
    return make


# ---------------------------------------------------------------------------
# СЦЕНАРИИ ДЛЯ BOEING 737-800
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ГРУППА 1. ДИАГНОСТИЧЕСКИЕ СЦЕНАРИИ
# ---------------------------------------------------------------------------
# Эти сценарии НЕ являются главным критерием валидности модели.
# Они нужны, чтобы показать границы применимости reduced-order модели.
# Например: high-speed horizontal flight, где модель может расходиться с JSBSim.

DIAGNOSTIC_SCENARIOS_737 = [
    Scenario(
        name="level_flight_trimmed_737",
        description=(
            "Диагностический high-speed горизонтальный полёт. "
            "Используется как sanity-check и демонстрация ограничений модели, "
            "но не как главный критерий валидации предсрывной динамики."
        ),
        config=CLEAN,
        t_final=90.0,
        make_initial_state=base_state(
            V=230.0,
            h=2000.0,
            theta_deg=-0.33,
            gamma_deg=0.0
        ),
        control_law=gamma_speed_controller(
            gamma_cmd_deg=0.0,
            throttle_trim=0.81269,
            V_cmd=230.0,
            u_trim=-0.10147
        ),
        group="diagnostic",
        purpose="high_speed_limit_check",
        primary_for_validation=False,
    ),
]


# ---------------------------------------------------------------------------
# ГРУППА 2. ВАЛИДАЦИОННЫЕ ПРЕДСРЫВНЫЕ TRIM-СЦЕНАРИИ
# ---------------------------------------------------------------------------
# Это самая важная группа.
# По этим сценариям проверяется, что модель может стартовать из физически
# осмысленного низкоскоростного режима:
#     dV ≈ 0, dgamma ≈ 0, dq ≈ 0
#
# Важно:
# theta_deg, throttle_trim, u_trim ниже — стартовые приближения.
# Их нужно заменить после подбора trim отдельно для JSBSim и отдельно для своей модели.

VALIDATION_TRIM_SCENARIOS_737 = [
    Scenario(
        name="clean_low_speed_trim_737",
        description=(
            "Валидационный низкоскоростной горизонтальный trim в чистой конфигурации. "
            "Основная задача — проверить локальное совпадение динамики с JSBSim "
            "в области, близкой к предсрывной."
        ),
        config=CLEAN,
        t_final=5.0,
        make_initial_state=base_state(
            V=110.0,
            h=2000.0,
            theta_deg=6.9249,
            gamma_deg=0.0
        ),
        control_law=gamma_speed_controller(
            gamma_cmd_deg=0.0,
            throttle_trim=0.3245,
            V_cmd=110.0,
            u_trim=-0.5058
        ),
        group="validation_trim",
        purpose="clean_low_speed_level_trim",
        primary_for_validation=True,
    ),

    Scenario(
        name="landing_low_speed_trim_737",
        description=(
            "Валидационный низкоскоростной trim в посадочной конфигурации. "
            "Используется для проверки предсрывной динамики при выпущенной посадочной конфигурации."
        ),
        config=LANDING,
        t_final=5.0,
        make_initial_state=base_state(
            V=95.0,
            h=800.0,
            theta_deg=0.8175,
            gamma_deg=0.0
        ),
        control_law=gamma_speed_controller(
            gamma_cmd_deg=0.0,
            throttle_trim=0.3156,
            V_cmd=95.0,
            u_trim=-0.1511
        ),
        group="validation_trim",
        purpose="landing_low_speed_level_trim",
        primary_for_validation=True,
    ),

    Scenario(
        name="low_speed_climb_trim_737",
        description=(
            "Валидационный trim набора высоты на малой скорости. "
            "Проверяет режим, где самолёт близок к энергетическому дефициту "
            "и может приблизиться к предсрывному состоянию."
        ),
        config=CLEAN,
        t_final=5.0,
        make_initial_state=base_state(
            V=115.0,
            h=1500.0,
            theta_deg=8.5726,
            gamma_deg=3.0
        ),
        control_law=gamma_speed_controller(
            gamma_cmd_deg=3.0,
            throttle_trim=0.4715,
            V_cmd=115.0,
            u_trim=-0.4304
        ),
        group="validation_trim",
        purpose="low_speed_climb_trim",
        primary_for_validation=True,
    ),
]


# ---------------------------------------------------------------------------
# ГРУППА 3. ПРЕДСРЫВНЫЕ / АВАРИЙНО-ВДОХНОВЛЁННЫЕ СЦЕНАРИИ
# ---------------------------------------------------------------------------
# Эти сценарии строятся на базе trim-сценариев.
# По ним оценивается не точное совпадение всей траектории, а event metrics:
#     t_alpha_8, t_alpha_10, t_alpha_12,
#     t_warning, t_stall,
#     min_speed_margin,
#     altitude_loss_before_recovery.

STALL_ENTRY_SCENARIOS_737 = [
    Scenario(
        name="clean_low_speed_to_stall_737",
        description=(
            "Вход в сваливание из низкоскоростного горизонтального полёта "
            "в чистой конфигурации: снижение тяги и постепенное увеличение тангажа."
        ),
        config=CLEAN,
        t_final=60.0,
        make_initial_state=base_state(
            V=110.0,
            h=2000.0,
            theta_deg=6.9249,   # должно совпадать с clean_low_speed_trim_737 после trim-подбора
            gamma_deg=0.0
        ),
        control_law=scheduled_trim_controller(
            schedule=[
                (0.0, 6.9249, 0.3245),
                (5.0, 7.5, 0.28),
                (12.0, 9.0, 0.22),
                (20.0, 11.0, 0.18),
                (28.0, 14.0, 0.15),
            ],
            u_trim=-0.5058,
            Kp=8.0,
            Kd=4.0
        ),
        group="stall_entry",
        purpose="clean_low_speed_stall_entry",
        parent_trim="clean_low_speed_trim_737",
        primary_for_validation=True,
    ),

    Scenario(
        name="landing_approach_to_stall_737",
        description=(
            "Вход в сваливание из посадочного низкоскоростного режима: "
            "уменьшение тяги и рост тангажа в посадочной конфигурации."
        ),
        config=LANDING,
        t_final=60.0,
        make_initial_state=base_state(
            V=95.0,
            h=800.0,
            theta_deg=0.8175,
            gamma_deg=0.0
        ),
        control_law=scheduled_trim_controller(
            schedule=[
                (0.0, 0.8175, 0.3156),
                (5.0, 0.8175, 0.3156),
                (12.0, 5.0, 0.22),
                (20.0, 8.0, 0.18),
                (28.0, 12.0, 0.15),
            ],
            u_trim=-0.1511,
            Kp=8.0,
            Kd=4.0
        ),
        group="stall_entry",
        purpose="landing_low_speed_stall_entry",
        parent_trim="landing_low_speed_trim_737",
        primary_for_validation=True,
    ),

    Scenario(
        name="low_speed_climb_to_stall_737",
        description=(
            "Предсрывной сценарий набора высоты на малой скорости: "
            "пилот пытается удерживать набор при недостаточной энергии, "
            "что приводит к росту угла атаки."
        ),
        config=CLEAN,
        t_final=60.0,
        make_initial_state=base_state(
            V=115.0,
            h=1500.0,
            theta_deg=8.5726,   # должно совпадать с low_speed_climb_trim_737 после trim-подбора
            gamma_deg=3.0
        ),
        control_law=scheduled_trim_controller(
            schedule=[
                (0.0, 8.5726, 0.4715),
                (8.0, 9.5, 0.42),
                (16.0, 11.0, 0.36),
                (24.0, 13.0, 0.32),
                (32.0, 16.0, 0.28),
            ],
            u_trim=-0.4304,
            Kp=8.0,
            Kd=4.0
        ),
        group="stall_entry",
        purpose="low_speed_climb_stall_entry",
        parent_trim="low_speed_climb_trim_737",
        primary_for_validation=True,
    ),

    Scenario(
        name="power_loss_stall_737",
        description=(
            "Аварийно-вдохновлённый сценарий: потеря тяги в низкоскоростном "
            "горизонтальном полёте и ошибочное увеличение тангажа."
        ),
        config=CLEAN,
        t_final=90.0,
        make_initial_state=base_state(
            V=110.0,
            h=2000.0,
            theta_deg=6.9249,
            gamma_deg=0.0
        ),
        control_law=gamma_speed_with_failure_controller(
            gamma_cmd_deg=0.0,
            throttle_trim=0.3245,
            V_cmd=110.0,
            u_trim=-0.5058,
            t_fail=5.0,
            fail_throttle=0.0,
            pitch_schedule=[
                (5.0, 6.9249, 0.0),
                (12.0, 8.0, 0.0),
                (18.0, 10.0, 0.0),
                (24.0, 12.0, 0.0),
                (30.0, 15.0, 0.0),
            ],
            Kp_gamma=8.0,
            Ki_gamma=0.3,
            Kd_gamma=2.0,
            Kp_V=0.005,
            Ki_V=0.0005,
            Kp_pitch=8.0,
            Kd_pitch=4.0
        ),
        group="stall_entry",
        purpose="power_loss_stall_entry",
        parent_trim="clean_low_speed_trim_737",
        primary_for_validation=True,
    ),

    Scenario(
        name="tk1951_landing_stall",
        description=(
            "Аварийно-вдохновлённый сценарий TK1951: посадочная конфигурация, "
            "уменьшение тяги, позднее восстановление. Используется как scenario-based stress test."
        ),
        config=LANDING,
        t_final=120.0,
        make_initial_state=base_state(
            V=80.0,
            h=300.0,
            theta_deg=0.8147,
            gamma_deg=-3.0
        ),
        control_law=tk1951_controller(
            gamma_cmd_deg=-3.0,
            throttle_trim=0.3631,
            V_cmd=80.0,
            u_trim=-0.1868,
            t_fail=15.0,
            fail_throttle=0.15,
            pitch_schedule=[
                (15.0, 10.0, 0.15),
                (25.0, 15.0, 0.15),
                (40.0, 20.0, 0.15),
            ],
            go_around_time=45.0,
            go_around_throttle=0.95,
            go_around_pitch_cmd_deg=5.0
        ),
        group="stall_entry",
        purpose="accident_inspired_landing_stall",
        parent_trim="landing_low_speed_trim_737",
        primary_for_validation=False,
    ),

    Scenario(
        name="af447_high_alt_stall",
        description=(
            "Аварийно-вдохновлённый high-altitude сценарий AF447. "
            "Используется как стресс-тест индекса риска, но не как главный критерий "
            "локальной валидации reduced-order модели."
        ),
        config=CLEAN,
        t_final=180.0,
        make_initial_state=base_state(
            V=230.0,
            h=10000.0,
            theta_deg=3.35,
            gamma_deg=0.0
        ),
        control_law=af447_controller(
            gamma_cmd_deg=0.0,
            throttle_trim=0.39168,
            V_cmd=230.0,
            u_trim=-0.30641,
            t_fail=10.0,
            cruise_throttle=0.15996,
            pitch_cmd_deg=5.0
        ),
        group="stall_entry",
        purpose="high_altitude_stress_test",
        parent_trim=None,
        primary_for_validation=False,
    ),

    Scenario(
        name="nax5630_elevator_jam",
        description=(
            "Аварийно-вдохновлённый сценарий заклинивания руля высоты. "
            "Важно: если отказ реализован только через ELEVATOR_EFFECT_MULTIPLIER "
            "в reduced-order модели, сценарий нельзя считать полноценным JSBSim-validation."
        ),
        config=CLEAN,
        t_final=120.0,
        make_initial_state=base_state(
            V=230.0,
            h=10000.0,
            theta_deg=3.35,
            gamma_deg=0.0
        ),
        control_law=elevator_jam_controller(
            gamma_cmd_deg=0.0,
            throttle_trim=0.39168,
            V_cmd=230.0,
            u_trim=-0.30641,
            t_jam=10.0,
            jam_elevator_cmd=-0.5,
            jam_effect_mult=0.1
        ),
        group="stall_entry",
        purpose="elevator_jam_stress_test",
        parent_trim=None,
        primary_for_validation=False,
    ),
]


# ---------------------------------------------------------------------------
# ЕДИНЫЙ СПИСОК СЦЕНАРИЕВ
# ---------------------------------------------------------------------------
# Сохраняем старое имя SCENARIOS_737, чтобы не ломать импорты:
# from experiments.scenarios_737 import SCENARIOS_737 as SCENARIOS

SCENARIOS_737 = (
    DIAGNOSTIC_SCENARIOS_737
    + VALIDATION_TRIM_SCENARIOS_737
    + STALL_ENTRY_SCENARIOS_737
)


# Удобный словарь групп для фильтрации в экспериментах и валидации
SCENARIO_GROUPS_737 = {
    "diagnostic": DIAGNOSTIC_SCENARIOS_737,
    "validation_trim": VALIDATION_TRIM_SCENARIOS_737,
    "stall_entry": STALL_ENTRY_SCENARIOS_737,
}


def get_scenarios_by_group(group_name: str):
    return SCENARIO_GROUPS_737.get(group_name, [])


def get_primary_validation_scenarios():
    return [
        scenario for scenario in SCENARIOS_737
        if scenario.primary_for_validation
    ]


def get_scenario_by_name(name: str):
    for scenario in SCENARIOS_737:
        if scenario.name == name:
            return scenario

    available = [scenario.name for scenario in SCENARIOS_737]
    raise ValueError(
        f"Scenario '{name}' not found. Available scenarios: {available}"
    )