"""
dynamics.py

Модель продольной динамики летательного аппарата.

Реализует упрощённую (reduced-order) систему уравнений движения:
- тангаж (θ)
- угловая скорость (q)
- скорость (V)
- высота (h)

Учитывает:
- аэродинамическую подъёмную силу
- сопротивление
- влияние управления (руль высоты, тяга)
- зависимость плотности воздуха от высоты
"""

import numpy as np
from config import min_v
from atmosphere import Atmosphere
from model.turbulence import TurbulenceModel
from config import STALL_WARNING_MARGIN_MPS
from dataclasses import dataclass
# Аэродимнамика: коэффициенты, угол атаки, векторы, отрыв потока

ELEVATOR_EFFECT_MULTIPLIER = 1.0
@dataclass
class AeroState:
    alpha: float
    rho: float
    sep: float
    dsep: float
    target_sep: float
    CL: float
    CD: float
    CL_max: float
    CL_ratio: float
    dCL_turb: float
    dCD_turb: float
    dCm_turb: float


# Силы
@dataclass
class ForceState:
    L: float
    D: float
    T: float
    T_x: float
    T_z: float
    X: float
    Z: float
    W: float


# Информация по сваливанию
@dataclass
class StallState:
    V_stall_1g: float
    V_stall_maneuver: float
    speed_margin_1g: float
    speed_margin_maneuver: float
    load_factor: float
    stall_warning_speed: float
    stall_warning_margin: float
    stall_warning: bool


# Моменты, коэффициенты, эффективность руля
@dataclass
class MomentState:
    Cm: float
    M: float
    q_dyn: float
    delta: float
    elevator_eff: float
    damping_eff: float

    Cm_base: float = 0.0
    Cm_config: float = 0.0
    Cm_alpha_part: float = 0.0
    Cm_q_part: float = 0.0
    Cm_delta_eff: float = 0.0
    Cm_delta_part: float = 0.0
    Cm_turb_part: float = 0.0

    speed_gain: float = 1.0
    config_gain: float = 1.0


# Параметры энергии
@dataclass
class EnergyState:
    vertical_speed: float
    power_available: float
    power_required_total: float
    excess_power: float
    energy_margin: float



class AircraftDynamics:
    """
    Класс динамики самолёта.
    Объединяет:
    - модель самолёта (AircraftModel)
    - атмосферу (Atmosphere)
    """

    def __init__(self, aircraft, config):
        self.aircraft = aircraft
        self.config = config
        self.atmosphere = Atmosphere()

        # гравитация
        self.g = 9.81

        self.turbulence = TurbulenceModel()

    # Аэродинамика

    def compute_alpha(self, theta, gamma):
        '''
        Функция расчета угла атаки
        '''
        alpha = theta - gamma

        return alpha

    def compute_CL(self, alpha, sep, V = None):
        """
        Коэффициент подъёмной силы с учётом post-stall режима.

        До сваливания:
            CL растёт почти линейно.
        После сваливания:
            CL переходит к плато и может частично сохраняться.
        """

        # коэффициент подъемной силы при нулевом угле атаки
        CL0 = self.aircraft.CL0 + self.config.dCL0 # базовый коэффициент подъемной силы с учетом положения закрылок
        CL_alpha = self.aircraft.CL_alpha # на сколько увеличится CL при изменении  угла атаки на 1 градус
        CL_max = self.config.CL_max if self.config.CL_max > 0 else self.aircraft.CL_max_clean

        # коэффициент подъемной силы до срыва потока
        CL_attached = CL0 + CL_alpha * alpha
        CL_attached = np.clip(CL_attached, -CL_max, CL_max)

        # Коэффициент подъемной силы после срыва потока
        sign = np.sign(alpha) if alpha != 0 else 1.0
        a = abs(alpha)

        # Остаточная подъемная сила после сваливания: когда крыло сорвалось, подъемная сила не падает до нуля резко
        CL_plateau = self.aircraft.CL_plateau_ratio * CL_max
        # Дополнительная подъемная сила от вихрей на больших углах атаки
        vortex = self.aircraft.CL_vortex_gain * np.sin(2.0 * a) ** 2

        CL_stalled = sign * (CL_plateau + vortex)

        # смешиваем нормальный и сорванный режимы
        CL = (1.0 - sep) * CL_attached + sep * CL_stalled

        return CL


    def compute_lift(self, rho, V, CL):
        """
        Подъёмная сила:

        L = 0.5 * ρ * V^2 * S * CL(α)
        """

        return 0.5 * rho * V**2 * self.aircraft.wing_area * CL

    def compute_CD(self, CL, alpha, sep):
        '''
        Функция расчета коэффициента сопротивления с учетом коэффициента подъемной силы
        '''
        Cd0 = self.aircraft.Cd0 + self.config.dCd0
        k = self.aircraft.k # Коэффициент индуктивного сопротивления

        # Чем больше подъемная сила, тем больше сопротивление
        CD_base = Cd0 + k * CL ** 2

        # дополнительное сопротивление после отрыва потока
        # 1. базровый рост сопротивления при сваливании
        # 2. зависимость сопротивления от угла атаки в срыве
        CD_stall = 0.35 * sep + 0.25 * sep * abs(np.sin(alpha))

        CD = CD_base + CD_stall

        return CD

    def compute_drag(self, rho, V, CD):
        '''
        Функция расчета силы сопротивления
        '''

        D = 0.5 * rho * V ** 2 * self.aircraft.wing_area * CD
        return D

    def pitch_damping_effectiveness(self, alpha, sep):
        """
        Эффективность демпфирования по тангажу.

        До сваливания:
            damping_eff ≈ 1

        При росте sep:
            аэродинамическое демпфирование ухудшается
        """

        alpha_stall = self.aircraft.alpha_stall_on + self.config.dAlpha_crit
        excess = max(0.0, abs(alpha) - alpha_stall)

        # ухудшение из-за превышения угла сваливания
        alpha_factor = np.exp(-2.0 * excess)

        # ухудшение из-за отрыва потока
        sep_factor = 1.0 - 0.7 * sep

        eff = alpha_factor * sep_factor

        return np.clip(eff, 0.2, 1.0)

    def elevator_effectiveness(self, alpha, sep):
        """
        Эффективность руля высоты.

        До сваливания ≈ 1.
        После роста sep эффективность падает.
        """

        alpha_stall = self.aircraft.alpha_stall_on + self.config.dAlpha_crit
        excess = max(0.0, abs(alpha) - alpha_stall)

        alpha_factor = np.exp(-1.5 * excess)
        sep_factor = 1.0 - 0.75 * sep

        eff = alpha_factor * sep_factor
        eff *= ELEVATOR_EFFECT_MULTIPLIER

        return np.clip(eff, 0.15, 1.0)

    def compute_sep_derivative(self, sep, alpha):
        '''
        Функция расчета срыва потока, к которому стремится текущий поток и dsep
        '''
        alpha_on = self.aircraft.alpha_stall_on + self.config.dAlpha_crit
        alpha_off = self.aircraft.alpha_stall_off + self.config.dAlpha_crit

        a = abs(alpha)

        # Подсчет нового срыва потока
        if a <= alpha_off:
            target_sep = 0.0
        elif a >= alpha_on:
            target_sep = 1.0
        else:
            x = (a - alpha_off) / max(alpha_on - alpha_off, 1e-6)
            target_sep = x * x * (3.0 - 2.0 * x)

        # Если поток, к которому стремится текущий поток больше текущего потока, используем формулу для срыва, иначе для восстановления
        # Выбор скорость изменения потока
        if target_sep > sep:
            tau = self.aircraft.tau_sep_stall
        else:
            tau = self.aircraft.tau_sep_recover

        dsep = (target_sep - sep) / tau

        return dsep, target_sep

    def compute_v_stall(self, rho):
        """
        Расчёт скорости сваливания для текущей конфигурации.

        Vstall = sqrt(2mg / (rho * S * CLmax))
        """

        CL_max = self.config.CL_max if self.config.CL_max > 0 else self.aircraft.CL_max_clean

        CL_max = max(CL_max, 1e-6)

        V_stall = np.sqrt(
            2.0 * self.aircraft.mass * self.g
            / (rho * self.aircraft.wing_area * CL_max)
        )

        return V_stall

#   для 737-800
    def compute_thrust(self, throttle, V, alpha):
        # Простая модель турбовентиляторного двигателя
        max_thrust = self.aircraft.static_thrust_max  # уже суммарная тяга двух двигателей
        T = throttle * max_thrust
        # Учёт угла атаки: проекция тяги на траекторию выполняется в compute_forces
        return T

#   для CESSNA_172S
#     def compute_thrust(self, throttle, V, alpha):
#         """
#         Простая модель винтовой тяги.
#
#         throttle — команда газа 0..1.
#         V — скорость полёта.
#         alpha — угол атаки.
#
#         Тяга ограничена мощностью двигателя и максимальной статической тягой.
#         """
#
#         throttle = np.clip(throttle, 0.0, 1.0) # Ограничение газа
#
#         P_available = throttle * self.aircraft.engine_power # Доступная мощность двигателя
#
#         # эффективность винта зависит от скорости
#         eta_prop = 0.6 + 0.4 * np.exp(-((V - 30.0) / 20.0) ** 2)
#         # Перевод мощности в тягу
#         T_power_limited = eta_prop * self.aircraft.prop_efficiency * P_available / max(V, 15.0)
#
#
#
#         # При увеличении угла атаки эффективность винта падает
#         alpha_abs = abs(alpha)
#         alpha_factor = 1.0 / (1.0 + (alpha_abs / np.radians(25.0)) ** 2)
#         alpha_factor = np.clip(alpha_factor, 0.25, 1.0)
#         # дополнительные потери эффективности винта при больших углах атаки
#         stall_drag_penalty = 1.0 - 0.6 * max(0.0, (alpha_abs - np.radians(10.0)) / np.radians(20.0))
#         stall_drag_penalty = np.clip(stall_drag_penalty, 0.3, 1.0)
#         T = T_power_limited * 1.2 / (1.0 + T_power_limited / (self.aircraft.static_thrust_max + 1e-6))
#         return T * alpha_factor * stall_drag_penalty

    def compute_aero_state(self, state, u, dt):
        '''
        Подсчет аэродинамики самолета в текущий момент

        '''


        theta = state.theta
        gamma = state.gamma
        h = state.h

        alpha = self.compute_alpha(theta, gamma)
        rho = self.atmosphere.density(h)

        dsep, target_sep = self.compute_sep_derivative(state.sep, alpha)
        sep = state.sep

        CL = self.compute_CL(alpha, sep)
        delta = u * self.aircraft.max_elevator
        #CL += self.aircraft.CL_delta_e * delta
        CD = self.compute_CD(CL, alpha, sep)
        CD += getattr(self.aircraft, "CD_delta_e", 0.0) * delta ** 2

        dCL_turb, dCD_turb, dCm_turb = self.turbulence.update(sep, dt)

        CL += dCL_turb

        CL_max = self.config.CL_max if self.config.CL_max > 0 else self.aircraft.CL_max_clean
        CL_ratio = CL / max(CL_max, 1e-6)

        CD += dCD_turb
        CD = max(CD, 0.01)

        return AeroState(
            alpha=alpha,
            rho=rho,
            sep=sep,
            dsep=dsep,
            target_sep=target_sep,
            CL=CL,
            CD=CD,
            CL_max=CL_max,
            CL_ratio=CL_ratio,
            dCL_turb=dCL_turb,
            dCD_turb=dCD_turb,
            dCm_turb=dCm_turb
        )

    def compute_forces(self, aero: AeroState, V, throttle):
        '''
        Функция расчета сил
        '''
        T = self.compute_thrust(throttle, V, aero.alpha)

        L = self.compute_lift(aero.rho, V, aero.CL)
        D = self.compute_drag(aero.rho, V, aero.CD)

        T_x = T * np.cos(aero.alpha) # Проекция на скорость/траекторию
        T_z = T * np.sin(aero.alpha) # Проекция на подъем

        X = T_x - D
        Z = L + T_z

        W = self.aircraft.mass * self.g

        return ForceState(
            L=L,
            D=D,
            T=T,
            T_x=T_x,
            T_z=T_z,
            X=X,
            Z=Z,
            W=W
        )


    def smoothstep(self, x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)


    def elevator_speed_gain(self, V):
        """
        Скоростная поправка эффективности руля высоты.

        На малых скоростях используем почти полную эффективность.
        На больших скоростях ослабляем, чтобы не получить чрезмерно резкую
        тангажную реакцию в high-speed режимах.
        """

        V_low = getattr(self.aircraft, "elevator_gain_v_low", 120.0)
        V_high = getattr(self.aircraft, "elevator_gain_v_high", 200.0)

        gain_low = getattr(self.aircraft, "elevator_gain_low", 1.0)
        gain_high = getattr(self.aircraft, "elevator_gain_high", 0.5)

        s = self.smoothstep((V - V_low) / max(V_high - V_low, 1e-6))

        return gain_low * (1.0 - s) + gain_high * s


    def compute_stall_margins(self, aero: AeroState, forces: ForceState, V):
        '''
        Расчет опасности по скорости сваливания с перегрузкой в 1g и в n
        '''
        # Расчет скорости сваливания при 1g. Это скорость при которой подъемная сила = весу самолета
        V_stall_1g = self.compute_v_stall(aero.rho)
        # Перегрузка (суммарная сила вверх (в траекторной системе)/вес самолета)
        load_factor = forces.Z / max(forces.W, 1e-6)

        # Маневренная скорость сваливания
        n_for_maneuver_stall = np.clip(max(load_factor, 1.0), 1.0, 4.0)
        V_stall_maneuver = V_stall_1g * np.sqrt(n_for_maneuver_stall)

        speed_margin_1g = V / max(V_stall_1g, 1e-6) # Запас по скорости относительно сваливания в 1g полете
        speed_margin_maneuver = V / max(V_stall_maneuver, 1e-6)

        # Для C172S логичнее предупреждение привязывать к 1g stall speed,
        # потому что POH-логика stall warning обычно формулируется относительно скорости сваливания,
        # а не относительно искусственно рассчитанной текущей манёвренной границы.
        stall_warning_speed = V_stall_1g + STALL_WARNING_MARGIN_MPS
        stall_warning_margin = V - stall_warning_speed
        stall_warning = V < stall_warning_speed

        return StallState(
            V_stall_1g=V_stall_1g,
            V_stall_maneuver=V_stall_maneuver,
            speed_margin_1g=speed_margin_1g,
            speed_margin_maneuver=speed_margin_maneuver,
            load_factor=load_factor,
            stall_warning_speed=stall_warning_speed,
            stall_warning_margin=stall_warning_margin,
            stall_warning=stall_warning
        )

    def compute_moments(self, aero: AeroState, state, u, V):
        q = state.q # угловая скорость вращения по тангажу

        q_dyn = 0.5 * aero.rho * V ** 2 # Сила потока воздуха на единицу плозади
        delta = u * self.aircraft.max_elevator

        elevator_eff = self.elevator_effectiveness(aero.alpha, aero.sep) # эффективность руля
        damping_eff = self.pitch_damping_effectiveness(aero.alpha, aero.sep) # Гашение самолетом вращения по тангажу

        q_hat = q * self.aircraft.mean_chord / (2 * V) # Нормированная угловая скорость

        Cm_base = self.aircraft.Cm0
        Cm_config = self.config.dCm0
        Cm_alpha_part = self.aircraft.Cm_alpha * aero.alpha
        Cm_q_part = self.aircraft.Cm_q * q_hat * damping_eff
        Cm_turb_part = aero.dCm_turb

        speed_gain = self.elevator_speed_gain(V)
        config_gain = getattr(self.config, "elevator_moment_gain", 1.0)

        Cm_delta_eff = (
                self.aircraft.Cm_delta_e
                * speed_gain
                * config_gain
        )

        Cm_delta_part = Cm_delta_eff * delta * elevator_eff

        Cm = (
                Cm_base
                + Cm_config
                + Cm_alpha_part
                + Cm_q_part
                + Cm_delta_part
                + Cm_turb_part
        )

        M = q_dyn * self.aircraft.wing_area * self.aircraft.mean_chord * Cm

        return MomentState(
            Cm=Cm,
            M=M,
            q_dyn=q_dyn,
            delta=delta,
            elevator_eff=elevator_eff,
            damping_eff=damping_eff,
            Cm_base=Cm_base,
            Cm_config=Cm_config,
            Cm_alpha_part=Cm_alpha_part,
            Cm_q_part=Cm_q_part,
            Cm_delta_eff=Cm_delta_eff,
            Cm_turb_part=Cm_turb_part,
        )

    def compute_energy_metrics(self, forces: ForceState, V, gamma, D, W):
        '''
        Функция проверки самолета на то, может ли он поддерживать текущий режим полета по энергии
        '''
        dh = V * np.sin(gamma)

        power_available = forces.T * V
        power_required_total = D * V - min(W * dh, 0.0)   # учёт помощи гравитации при снижении
        excess_power = power_available - power_required_total

        energy_margin = excess_power / max(power_required_total, 1.0)

        return EnergyState(
            vertical_speed=dh,
            power_available=power_available,
            power_required_total=power_required_total,
            excess_power=excess_power,
            energy_margin=energy_margin
        )

    def write_diagnostics_to_state(
            self,
            state,
            aero: AeroState,
            forces: ForceState,
            stall: StallState,
            moment: MomentState,
            energy: EnergyState,
            throttle
    ):
        state.alpha = aero.alpha
        state.sep_target = aero.target_sep

        state.CL = aero.CL
        state.CD = aero.CD
        state.CL_max = aero.CL_max
        state.CL_ratio = aero.CL_ratio

        state.dCL_turb = aero.dCL_turb
        state.dCD_turb = aero.dCD_turb
        state.dCm_turb = aero.dCm_turb

        state.throttle = throttle
        state.thrust = forces.T

        state.load_factor = stall.load_factor

        state.V_stall_1g = stall.V_stall_1g
        state.V_stall_maneuver = stall.V_stall_maneuver

        state.speed_margin_1g = stall.speed_margin_1g
        state.speed_margin_maneuver = stall.speed_margin_maneuver

        # Совместимость со старым логом/визуализацией
        state.V_stall = stall.V_stall_maneuver
        state.speed_margin = stall.speed_margin_maneuver

        state.stall_warning_speed = stall.stall_warning_speed
        state.stall_warning_margin = stall.stall_warning_margin
        state.stall_warning = stall.stall_warning

        state.Cm = moment.Cm
        state.elevator_eff = moment.elevator_eff
        state.pitch_damping_eff = moment.damping_eff
        state.elevator_delta = moment.delta
        state.elevator_speed_gain = moment.speed_gain
        state.elevator_config_gain = moment.config_gain
        state.Cm_delta_eff = moment.Cm_delta_eff
        state.Cm_delta_part = moment.Cm_delta_part

        state.vertical_speed = energy.vertical_speed
        state.power_available = energy.power_available
        state.power_required_total = energy.power_required_total
        state.excess_power = energy.excess_power
        state.energy_margin = energy.energy_margin

    def compute_diagnostics(self, state, u, throttle, dt):
        theta = state.theta
        q = state.q
        V = max(state.V, min_v)
        gamma = state.gamma

        aero = self.compute_aero_state(state, u, dt)
        forces = self.compute_forces(aero, V, throttle)
        stall = self.compute_stall_margins(aero, forces, V)
        moment = self.compute_moments(aero, state, u, V)
        energy = self.compute_energy_metrics(forces, V, gamma, forces.D, forces.W)

        return aero, forces, stall, moment, energy
    # Динамика

    def derivatives(self, state, u, throttle, dt):
        theta = state.theta
        q = state.q
        V = max(state.V, min_v)
        gamma = state.gamma

        aero = self.compute_aero_state(state, u, dt)
        forces = self.compute_forces(aero, V, throttle)
        stall = self.compute_stall_margins(aero, forces, V)
        moment = self.compute_moments(aero, state, u, V)
        energy = self.compute_energy_metrics(forces, V, gamma, forces.D, forces.W)

        # self.write_diagnostics_to_state(
        #     state=state,
        #     aero=aero,
        #     forces=forces,
        #     stall=stall,
        #     moment=moment,
        #     energy=energy,
        #     throttle=throttle
        # )

        dtheta = q
        dq = moment.M / self.aircraft.Iyy

        dV = (
                forces.X / self.aircraft.mass
                - self.g * np.sin(gamma)
        )
        dgamma = (
                (forces.Z - self.aircraft.mass * self.g * np.cos(gamma))
                / (self.aircraft.mass * V)
        )

        dh = energy.vertical_speed
        dx = V * np.cos(gamma)
        diagnostics = (
            aero,
            forces,
            stall,
            moment,
            energy,
            throttle
        )

        return dtheta, dq, dV, dgamma, dh, dx, aero.dsep, diagnostics
