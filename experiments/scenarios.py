# experiments/scenarios.py


from dataclasses import dataclass
from typing import Callable
import numpy as np

from AircraftModel import BOEING_737_800, CLEAN, LANDING
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


def clamp_control(u):
    return float(np.clip(u, -1.0, 1.0))


def pitch_hold_controller(theta_cmd_deg, throttle_cmd, Kp=1.2, Kd=0.5):
    theta_cmd_rad = np.radians(theta_cmd_deg)
    def control(t, state):
        u = Kp * (theta_cmd_rad - state.theta) - Kd * state.q
        return clamp_control(u), throttle_cmd
    return control


def scheduled_pitch_controller(schedule, Kp=1.2, Kd=0.5):
    """
    schedule: список кортежей (t_start, theta_cmd_deg, throttle)
    Берётся последняя команда, у которой t >= t_start.
    """
    schedule = sorted(schedule, key=lambda x: x[0])

    def control(t, state):
        theta_cmd_deg = schedule[0][1]
        throttle = schedule[0][2]

        for t_start, theta_deg, thr in schedule:
            if t >= t_start:
                theta_cmd_deg = theta_deg
                throttle = thr
            else:
                break

        theta_cmd = np.radians(theta_cmd_deg)
        u = Kp * (theta_cmd - state.theta) - Kd * state.q
        return clamp_control(u), throttle

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



SCENARIOS = [
    Scenario(
        name="normal_flight_clean",
        description="Нормальный полёт в чистой конфигурации без приближения к сваливанию.",
        config=CLEAN,
        t_final=20.0,
        make_initial_state=base_state(V=55.0, h=300.0, theta_deg=4.0, gamma_deg=2.0),
        control_law=pitch_hold_controller(5.0, throttle_cmd=0.55),
    ),

    Scenario(
        name="low_speed_clean",
        description="Плавный переход к малой скорости при умеренном тангаже и малой тяге.",
        config=CLEAN,
        t_final=25.0,
        make_initial_state=base_state(V=42.0, h=300.0, theta_deg=8.0, gamma_deg=3.0),
        control_law=pitch_hold_controller(15.0, throttle_cmd=0.30),
    ),

    Scenario(
        name="clean_power_off_stall",
        description="Сваливание в чистой конфигурации при малой/нулевой тяге и постепенном увеличении тангажа.",
        config=CLEAN,
        t_final=25.0,
        make_initial_state=base_state(V=50.0, h=500.0, theta_deg=4.0, gamma_deg=2.0),
        control_law=scheduled_pitch_controller([
            (0.0, 8.0, 0.00),
            (4.0, 14.0, 0.00),
            (8.0, 20.0, 0.00),
            (12.0, 26.0, 0.00),
        ]),
    ),

    Scenario(
        name="clean_power_on_stall",
        description="Сваливание в чистой конфигурации при повышенной тяге и увеличении тангажа.",
        config=CLEAN,
        t_final=25.0,
        make_initial_state=base_state(V=45.0, h=500.0, theta_deg=6.0, gamma_deg=3.0),
        control_law=scheduled_pitch_controller([
            (0.0, 10.0, 0.70),
            (4.0, 16.0, 0.75),
            (8.0, 22.0, 0.75),
            (12.0, 28.0, 0.75),
        ]),
    ),

    Scenario(
        name="stall_recovery_clean",
        description="Вход в сваливание в чистой конфигурации с последующим снижением тангажа и увеличением тяги.",
        config=CLEAN,
        t_final=30.0,
        make_initial_state=base_state(V=48.0, h=600.0, theta_deg=6.0, gamma_deg=4.0),
        control_law=scheduled_pitch_controller([
            (0.0, 24.0, 0.25),
            (7.0, 2.0, 0.85),
            (14.0, 6.0, 0.60),
        ]),
    ),

    Scenario(
        name="low_speed_climb",
        description="Набор высоты на малой скорости: проверка энергетического дефицита и приближения к опасному режиму.",
        config=CLEAN,
        t_final=30.0,
        make_initial_state=base_state(V=34.0, h=300.0, theta_deg=10.0, gamma_deg=5.0),
        control_law=pitch_hold_controller(16.0, throttle_cmd=0.55),
    ),

    Scenario(
        name="landing_config_stall",
        description="Сваливание в посадочной конфигурации с выпущенными закрылками.",
        config=LANDING,
        t_final=25.0,
        make_initial_state=base_state(V=38.0, h=500.0, theta_deg=5.0, gamma_deg=2.0),
        control_law=scheduled_pitch_controller([
            (0.0, 8.0, 0.20),
            (4.0, 12.0, 0.20),
            (8.0, 16.0, 0.20),
            (12.0, 20.0, 0.20),
        ]),
    ),
    Scenario(
        name="accelerated_pull_up",
        description="Ускоренное сваливание при резком увеличении тангажа и росте перегрузки.",
        config=CLEAN,
        t_final=20.0,
        make_initial_state=base_state(V=55.0, h=500.0, theta_deg=3.0, gamma_deg=0.0),
        control_law=scheduled_pitch_controller([
            (0.0, 5.0, 0.55),
            (3.0, 18.0, 0.55),
            (6.0, 28.0, 0.55),
        ]),
    ),

]