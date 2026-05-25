"""
analytic_validation.py

Аналитическая верификация 3‑DOF модели Boeing 737‑800.
Адаптировано с версии для Cessna 172S.
"""

import numpy as np
from AircraftModel import BOEING_737_JSBSIM, CLEAN, LANDING
from model.dynamics import AircraftDynamics
from model.state import State
from atmosphere import Atmosphere
from scipy.optimize import minimize

G = 9.81
DT = 0.0001

TOL_FORCE = 100.0      # допустимое отклонение сил, Н (увеличено для B738)
TOL_MOMENT = 1000.0     # допустимое отклонение момента, Н·м
H = 500.0
def make_tmp_state(alpha, gamma, V, h=H):
    s = State()
    s.theta = alpha + gamma
    s.gamma = gamma
    s.V = V
    s.h = h
    s.q = 0.0
    s.x = 0.0
    s.R = 0.0
    s.sep = 0.0
    return s


def unpack_derivatives(result):
    """
    Поддерживает обе версии dynamics.derivatives():
    - старая: dtheta, dq, dV, dgamma, dh, dx, dsep
    - новая: dtheta, dq, dV, dgamma, dh, dx, dsep, diagnostics
    """

    dtheta = result[0]
    dq = result[1]
    dV = result[2]
    dgamma = result[3]
    dh = result[4]
    dx = result[5]
    dsep = result[6]

    diagnostics = result[7] #if len(result) > 7 else None

    return dtheta, dq, dV, dgamma, dh, dx, dsep, diagnostics


def find_alpha_for_CL(aircraft, config, CL_target, V, gamma=0.0, u=0.0):
    """
    Грубый поиск alpha для заданного CL при фиксированном u.

    В новой модели CL зависит не только от alpha, но и от elevator:
        CL = CL(alpha, delta_e)

    Поэтому u обязательно передаётся в compute_aero_state.
    Эту функцию лучше использовать только для начального приближения.
    """

    dynamics = AircraftDynamics(aircraft, config)
    alpha = 0.05

    for _ in range(100):
        state = make_tmp_state(alpha, gamma, V)

        aero = dynamics.compute_aero_state(
            state=state,
            u=u,
            dt=DT
        )

        CL = aero.CL

        if abs(CL - CL_target) < 1e-6:
            break

        state2 = make_tmp_state(alpha + 0.005, gamma, V)

        aero2 = dynamics.compute_aero_state(
            state=state2,
            u=u,
            dt=DT
        )

        dCL_dalpha = (aero2.CL - CL) / 0.005

        if abs(dCL_dalpha) < 1e-9:
            break

        alpha += (CL_target - CL) / dCL_dalpha

    return alpha


def find_u_for_Cm_zero(aircraft, config, alpha, V, gamma=0.0, throttle=0.0, h=H):
    """
    Подбирает u, при котором Cm около нуля, при фиксированных alpha, V, gamma.

    Используется только как начальное приближение.
    """

    dynamics = AircraftDynamics(aircraft, config)

    def cost_u(u_array):
        u = float(u_array[0])
        state = make_tmp_state(alpha, gamma, V, h)

        aero, forces, stall, moment, energy = dynamics.compute_diagnostics(
            state=state,
            u=u,
            throttle=throttle,
            dt=DT
        )

        return moment.Cm ** 2

    res = minimize(
        cost_u,
        x0=[0.0],
        bounds=[(-1.0, 1.0)],
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-14}
    )

    if res.success:
        return float(np.clip(res.x[0], -1.0, 1.0))

    return 0.0


def compute_balance(aircraft, config, alpha, gamma, V, throttle, u, h=H):
    dynamics = AircraftDynamics(aircraft, config)
    state = make_tmp_state(alpha, gamma, V, h)
    state.elevator_delta = u * aircraft.max_elevator
    aero, forces, stall, moment, energy = dynamics.compute_diagnostics(
        state, u=u, throttle=throttle, dt=DT
    )
    return forces.L, forces.D, forces.T, moment.M, aero, moment

def test_steady_glide():
    """
    Подбор установившегося планирования:
    throttle фиксирован = 0,
    оптимизируются theta, gamma, u.

    Условие установившегося режима:
        dV ≈ 0
        dgamma ≈ 0
        dq ≈ 0
    """

    print("\n========== Тест 1: Установившееся планирование ==========")

    aircraft = BOEING_737_JSBSIM
    config = CLEAN
    dynamics = AircraftDynamics(aircraft, config)

    W = aircraft.mass * G
    V = 80.0
    h = H
    throttle = 0.0

    # Начальные приближения.
    gamma0 = np.radians(-3.0)

    rho = Atmosphere().density(h)
    q_dyn = 0.5 * rho * V ** 2
    CL_req = W * np.cos(gamma0) / max(q_dyn * aircraft.wing_area, 1e-9)

    CL0 = aircraft.CL0 + config.dCL0
    alpha0 = (CL_req - CL0) / max(aircraft.CL_alpha, 1e-9)
    alpha0 = np.clip(alpha0, np.radians(-5.0), np.radians(12.0))

    theta0 = alpha0 + gamma0

    u0 = find_u_for_Cm_zero(
        aircraft=aircraft,
        config=config,
        alpha=alpha0,
        V=V,
        gamma=gamma0,
        throttle=throttle,
        h=h
    )

    x0 = np.array([
        theta0,
        gamma0,
        u0,
    ])

    bounds = [
        (np.radians(-15.0), np.radians(15.0)),   # theta
        (np.radians(-15.0), np.radians(2.0)),    # gamma
        (-1.0, 1.0),                             # u
    ]

    def cost(params):
        theta, gamma, u = params
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
            throttle=throttle,
            dt=DT
        )

        _, dq, dV, dgamma, _, _, _, _ = unpack_derivatives(result)

        # Масштабируем, чтобы optimizer не игнорировал малые угловые величины.
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
        options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 1000}
    )

    if not res.success:
        print("⚠ Оптимизация не сошлась, используются лучшие найденные значения.")
        print(f"Причина: {res.message}")

    theta_opt, gamma_opt, u_opt = res.x
    alpha_opt = theta_opt - gamma_opt

    L, D, T, M, aero, moment_state = compute_balance(
        aircraft,
        config,
        alpha_opt,
        gamma_opt,
        V,
        throttle=throttle,
        u=u_opt,
        h=h
    )

    state_test = make_tmp_state(alpha_opt, gamma_opt, V, h)
    result = dynamics.derivatives(
        state_test,
        u=float(u_opt),
        throttle=throttle,
        dt=DT
    )

    _, dq, dV, dgamma, _, _, _, _ = unpack_derivatives(result)

    print(f"Найден режим планирования:")
    print(f"  V = {V:.2f} м/с, h = {h:.1f} м")
    print(f"  gamma = {np.degrees(gamma_opt):+.3f}°")
    print(f"  theta = {np.degrees(theta_opt):+.3f}°")
    print(f"  alpha = {np.degrees(alpha_opt):+.3f}°")
    print(f"  throttle = {throttle:.5f}")
    print(f"  u = {u_opt:+.5f}")
    print(f"  delta_e = {np.degrees(u_opt * aircraft.max_elevator):+.3f}°")
    print()
    print(f"  L = {L:.1f} Н (W cosγ = {W*np.cos(gamma_opt):.1f} Н)")
    print(f"  D = {D:.1f} Н (-W sinγ = {-W*np.sin(gamma_opt):.1f} Н)")
    print(f"  T = {T:.1f} Н")
    print(f"  M = {M:.1f} Н·м")
    print(f"  CL = {aero.CL:.5f}, CD = {aero.CD:.5f}, Cm = {moment_state.Cm:.6f}")
    print()
    print(f"  Производные:")
    print(f"    dV     = {dV:+.8f} м/с²")
    print(f"    dgamma = {dgamma:+.8f} рад/с")
    print(f"    dq     = {dq:+.8f} рад/с²")

    passed = (
        abs(dV) < 1e-3
        and abs(dgamma) < 1e-5
        and abs(dq) < 1e-5
    )

    print(f"Тест {'пройден' if passed else 'не пройден'}")

    return passed


def test_trim_flight(aircraft, config, V, h, gamma_deg=0.0):
    """
    Универсальный тест балансировки для любых V, h, γ, конфигурации.
    Возвращает passed (bool).
    """
    print(f"\n========== Балансировка: V={V:.1f} м/с, h={h:.0f} м, γ={gamma_deg:.1f}° ==========")
    dynamics = AircraftDynamics(aircraft, config)
    W = aircraft.mass * G
    gamma = np.radians(gamma_deg)

    # Автоматический подбор начальных приближений
    # Оценим потребный угол атаки из условия L ≈ W cos γ
    rho = Atmosphere().density(h)
    q = 0.5 * rho * V**2
    CL_req = W * np.cos(gamma) / (q * aircraft.wing_area)
    # Грубая оценка α по линейной модели
    CL0 = aircraft.CL0 + config.dCL0
    alpha_est = (CL_req - CL0) / aircraft.CL_alpha
    theta0 = alpha_est + gamma

    # Начальная тяга: уравновесить сопротивление (приближённо)
    CD_est = aircraft.Cd0 + config.dCd0 + aircraft.k * CL_req**2
    D_est = q * aircraft.wing_area * CD_est
    throttle0 = np.clip(D_est / aircraft.static_thrust_max, 0.05, 0.95)

    # Руль высоты: примерно убрать момент тангажа
    u0 = 0.0

    # Широкие границы, чтобы оптимизатор мог найти решение
    bounds = [
        (theta0 - np.radians(10), theta0 + np.radians(10)),  # θ
        (0.0, 1.0),                                           # throttle
        (-1.0, 1.0)                                           # u
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

    res = minimize(cost, [theta0, throttle0, u0], bounds=bounds,
                   method='L-BFGS-B', options={'ftol': 1e-12, 'gtol': 1e-12})

    if not res.success:
        print("⚠ Оптимизация не сошлась, используются начальные приближения.")
        theta_opt, throttle_opt, u_opt = theta0, throttle0, u0
    else:
        theta_opt, throttle_opt, u_opt = res.x
        print(f"Оптимизация сошлась за {res.nit} итераций.")

    alpha_opt = theta_opt - gamma

    L, D, T, M, aero, moment_state = compute_balance(
        aircraft, config, alpha_opt, gamma, V, throttle_opt, u_opt, h
    )

    print(f"  α = {np.degrees(alpha_opt):.2f}°, θ = {np.degrees(theta_opt):.2f}°")
    print(f"  throttle = {throttle_opt:.5f}, u = {u_opt:.5f}  (δe = {u_opt*aircraft.max_elevator:.4f} рад)")
    print(f"  L = {L:.1f} Н (треб. {W*np.cos(gamma):.1f} Н)")
    print(f"  D = {D:.1f} Н, T = {T:.1f} Н")
    print(f"  M = {M:.1f} Н·м")

    state_test = make_tmp_state(alpha_opt, gamma, V, h)
    state_test.elevator_delta = u_opt * aircraft.max_elevator
    result = dynamics.derivatives(
        state_test,
        u=float(u_opt),
        throttle=float(throttle_opt),
        dt=DT
    )

    _, dq, dV, dgamma, _, _, _, _ = unpack_derivatives(result)
    print(f"  Производные: dV={dV:.6f}, dgamma={dgamma:.6f}, dq={dq:.6f}")

    passed = (abs(L - W*np.cos(gamma)) < TOL_FORCE and
              abs(T - D) < TOL_FORCE and
              abs(M) < TOL_MOMENT and
              abs(dV) < 1e-3 and abs(dgamma) < 1e-5 and abs(dq) < 1e-5)
    print(f"Тест {'пройден' if passed else 'не пройден'}")
    return passed


def test_free_fall():
    print("\n========== Тест 3: Свободное падение ==========")
    aircraft = BOEING_737_JSBSIM
    config = CLEAN
    gamma = -np.pi/2
    V = 0.1
    state = make_tmp_state(alpha=0.0, gamma=gamma, V=V)
    state.theta = gamma
    dynamics = AircraftDynamics(aircraft, config)
    aero, forces, stall, moment, energy = dynamics.compute_diagnostics(
        state, u=0.0, throttle=0.0, dt=DT
    )
    dV_dt_expected = G
    dV_dt_actual = -G * np.sin(gamma)
    print(f"  dV/dt ожидаемое = {dV_dt_expected:.2f} м/с²")
    print(f"  dV/dt расчётное = {dV_dt_actual:.2f} м/с²")
    passed = abs(dV_dt_actual - dV_dt_expected) < 0.01
    print(f" Тест {'пройден' if passed else 'не пройден'}")
    return passed


# if __name__ == "__main__":
#     results = []
#     results.append(("Установившееся планирование", test_steady_glide()))
#     results.append(("Горизонтальный полёт", test_level_flight()))
#     results.append(("Свободное падение", test_free_fall()))
#     print("\n========== Итог ==========")
#     all_ok = all(r[1] for r in results)
#     for name, ok in results:
#         print(f"{'✅' if ok else '❌'} {name}")
#     print(f"\nОбщий результат: {'Все тесты пройдены' if all_ok else 'Обнаружены расхождения'}")


if __name__ == "__main__":
    results = []
    results.append(("Установившееся планирование", test_steady_glide()))

    # 1) Горизонтальный полёт CLEAN на H=2000 м, V=230 м/с (для level_flight_trimmed_737)
    results.append(("Гориз. полёт CLEAN 230 м/с, 2000 м",
                    test_trim_flight(BOEING_737_JSBSIM, CLEAN, V=230.0, h=2000.0, gamma_deg=0.0)))
    # 2) Отказ двигателя CLEAN на H=2000 м, V=150 м/с (для level_flight_trimmed_737)
    results.append(("Гориз. полет CLEAN 150 м/с, 2000 м",
                    test_trim_flight(BOEING_737_JSBSIM, CLEAN, V=150.0, h=2000.0, gamma_deg=0.0)))

    # 3) Снижение LANDING V=70 м/с, H=300 м, γ=-3° (для TK1951)
    results.append(("Снижение LANDING 80 м/с, 300 м, γ=-3°",
                    test_trim_flight(BOEING_737_JSBSIM, LANDING, V=80.0, h=300.0, gamma_deg=-3.0)))

    # 4) Горизонтальный полёт CLEAN на H=10000 м, V=230 м/с (для NAX5630 и AF447)
    results.append(("Гориз. полёт CLEAN 230 м/с, 10000 м",
                    test_trim_flight(BOEING_737_JSBSIM, CLEAN, V=230.0, h=10000.0, gamma_deg=0.0)))

    results.append(("Свободное падение", test_free_fall()))

    print("\n========== Итог ==========")
    all_ok = all(r[1] for r in results)
    for name, ok in results:
        print(f"{'✅' if ok else '❌'} {name}")
    print(f"\nОбщий результат: {'Все тесты пройдены' if all_ok else 'Обнаружены расхождения'}")