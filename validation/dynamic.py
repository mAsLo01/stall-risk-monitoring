"""
dynamic.py

Валидационные тесты для 3‑DOF модели самолёта.
Проверяют:
1) закон сохранения энергии при планировании,
2) закон сохранения импульса вдоль траектории,
3) закон сохранения момента импульса (вращательная динамика).
"""

import numpy as np
from simulation.rk2 import RK2Integrator
from model.dynamics import AircraftDynamics
from model.risk import RiskModel
from model.state import State
from AircraftModel import BOEING_737_800, CLEAN

G = 9.81

def validate_energy_conservation(aircraft, config, V0, h0, gamma0, theta0, u, dt, t_max):
    """
    Тест 1: Закон сохранения энергии при планировании.
    ΔE_mech должен равняться работе силы сопротивления (с обратным знаком).
    """
    dynamics = AircraftDynamics(aircraft, config)
    integrator = RK2Integrator()
    risk = RiskModel()

    state = State()
    state.V = V0
    state.h = h0
    state.gamma = gamma0
    state.theta = theta0
    state.q = 0.0
    state.sep = 0.0

    t = 0.0
    E_mech0 = aircraft.mass * G * h0 + 0.5 * aircraft.mass * V0**2
    work_drag = 0.0

    while t < t_max:
        state = integrator.step(state, dynamics, risk, u, throttle=0.0, dt=dt)
        # Получаем диагностику сил
        aero, forces, stall, moment, energy = dynamics.compute_diagnostics(state, u, 0.0, dt)
        work_drag += forces.D * state.V * dt
        t += dt
        if state.h < 0:
            break

    E_mech_final = aircraft.mass * G * state.h + 0.5 * aircraft.mass * state.V**2
    delta_E = E_mech_final - E_mech0
    error = abs(delta_E + work_drag) / max(abs(E_mech0), 1.0)
    print(f"  ΔE = {delta_E:.1f} Дж, работа D = {work_drag:.1f} Дж, относительная ошибка = {error:.6f}")
    return error < 0.01


def validate_impulse_conservation(aircraft, config, V0, h0, gamma0, theta0, u, throttle, dt, t_max):
    """
    Тест 2: Закон сохранения импульса вдоль траектории.
    m * (Vx_final - Vx_initial) должно равняться интегралу от (T_x - D) dt.
    """
    dynamics = AircraftDynamics(aircraft, config)
    integrator = RK2Integrator()
    risk = RiskModel()

    state = State()
    state.V = V0
    state.h = h0
    state.gamma = gamma0
    state.theta = theta0
    state.q = 0.0
    state.sep = 0.0

    Vx0 = V0 * np.cos(gamma0)
    integral_net_force = 0.0
    t = 0.0

    while t < t_max:
        state = integrator.step(state, dynamics, risk, u, throttle=throttle, dt=dt)
        aero, forces, stall, moment, energy = dynamics.compute_diagnostics(state, u, throttle, dt)
        net_force = forces.T * np.cos(aero.alpha) - forces.D   # T_x - D
        integral_net_force += net_force * dt
        t += dt
        if state.h < 0:
            break

    Vx_final = state.V * np.cos(state.gamma)
    delta_p = aircraft.mass * (Vx_final - Vx0)
    error = abs(delta_p - integral_net_force) / max(abs(aircraft.mass * Vx0), 1.0)
    print(f"  Δ импульса = {delta_p:.1f} Н·с, интеграл (T_x - D) = {integral_net_force:.1f} Н·с, отн. ошибка = {error:.6f}")
    return error < 0.01


def validate_angular_momentum(aircraft, config, V0, h0, gamma0, theta0, u, throttle, dt, t_max):
    dynamics = AircraftDynamics(aircraft, config)
    integrator = RK2Integrator()
    risk = RiskModel()

    state = State()
    state.V = V0
    state.h = h0
    state.gamma = gamma0
    state.theta = theta0
    state.q = 0.0
    state.sep = 0.0

    integral_M = 0.0
    t = 0.0

    while t < t_max:
        # Получаем производные, включая момент, ДО шага
        dtheta, dq, dV, dgamma, dh, dx, dsep, diag = dynamics.derivatives(
            state, u, throttle, dt
        )
        # diag содержит (aero, forces, stall, moment, energy, throttle)
        moment = diag[3]                     # MomentState
        M_current = moment.M                 # момент, который будет приложен
        integral_M += M_current * dt

        # Выполняем шаг (внутри integrator.step снова вызовет derivatives, но это нормально)
        state = integrator.step(state, dynamics, risk, u, throttle, dt)

        t += dt
        if state.h < 0:
            break

    delta_L = aircraft.Iyy * state.q
    abs_error = abs(delta_L - integral_M)
    # Характерный момент для Boeing 737: q_dyn * S * c * Cmδe * δe_max ≈ 5000 Н·м·с (за 2 с)
    characteristic_M_impulse = 10000.0  # Н·м·с, можно вычислить точнее
    passed = abs_error < 0.01 * characteristic_M_impulse
    print(f"  Абсолютная невязка = {abs_error:.2f} Н·м·с (порог {0.01 * characteristic_M_impulse:.2f} Н·м·с)")
    return passed

if __name__ == "__main__":
    print("=== Валидация на основе законов сохранения (Boeing 737-800) ===\n")

    aircraft = BOEING_737_800
    config = CLEAN
    dt = 0.06          # шаг симуляции (как в config.py)
    t_max = 5.0        # длительность каждого теста, сек

    # --- Параметры для Теста 1 (планирование) ---
    # Берём из аналитического теста установившегося планирования:
    V0_glide = 80.0
    h0_glide = 2000.0
    gamma0_glide = np.radians(-4.47)   # из вашего теста для 2000 м
    alpha_glide = np.radians(9.81)     # примерно 9.8° (из теста CL=1.856)
    theta0_glide = gamma0_glide + alpha_glide
    u_glide = -0.2608                  # из аналитического теста

    # --- Параметры для Теста 2 (горизонтальный полёт) ---
    # Из оптимизированного теста горизонтального полёта:
    V0_level = 230.0
    h0_level = 2000.0
    gamma0_level = 0.0
    theta0_level = np.radians(-1.09)
    u_level = -0.1106
    throttle_level = 0.2517

    # --- Параметры для Теста 3 (возмущение момента) ---
    # Используем те же, что в горизонтальном полёте, но подадим
    # кратковременный импульс рулём, чтобы проверить момент.
    u_disturb = -0.5   # отклоним руль сильнее на 2 секунды

    print("Тест 1: Закон сохранения энергии (планирование 5 с)")
    res1 = validate_energy_conservation(aircraft, config, V0_glide, h0_glide,
                                         gamma0_glide, theta0_glide, u_glide, dt, t_max)
    print(f"  Результат: {'✅ пройден' if res1 else '❌ не пройден'}\n")

    print("Тест 2: Закон сохранения импульса (горизонтальный полёт 5 с)")
    res2 = validate_impulse_conservation(aircraft, config, V0_level, h0_level,
                                          gamma0_level, theta0_level, u_level, throttle_level, dt, t_max)
    print(f"  Результат: {'✅ пройден' if res2 else '❌ не пройден'}\n")

    print("Тест 3: Закон сохранения момента импульса (возмущение 2 с, затем стабилизация)")
    # Сначала даём возмущение 2 секунды, потом возвращаем равновесное управление
    # Для простоты можно просто 2 секунды подать u_disturb, а потом обнулить.
    # Но мы уже написали универсальную функцию, использующую постоянное u.
    # Сделаем два вызова: первый с возмущением, второй без.
    # Для наглядности здесь просто запустим с u_disturb на 2 секунды.
    res3 = validate_angular_momentum(aircraft, config, V0_level, h0_level,
                                      gamma0_level, theta0_level, u_disturb, throttle_level, dt, 2.0)
    print(f"  Результат: {'✅ пройден' if res3 else '❌ не пройден'}\n")

    print("=== Итог ===")
    all_ok = res1 and res2 and res3
    print(f"{'Все тесты пройдены' if all_ok else 'Обнаружены расхождения'}")