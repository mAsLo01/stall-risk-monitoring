# import numpy as np
#
# class StallAeroModel:
#     def __init__(self, aircraft):
#         self.aircraft = aircraft
#
#         # динамическое состояние срыва потока
#         self.sep = 0.0          # 0 = поток присоединён, 1 = глубокий срыв
#         self.alpha_eff = 0.0    # запаздывающий угол атаки
#
#     def update_dynamic_states(self, alpha, q_hat, dt):
#         ac = self.aircraft
#
#         # запаздывание эффективного угла атаки
#         tau_alpha = ac.tau_alpha
#         self.alpha_eff += dt * (alpha - self.alpha_eff) / tau_alpha
#
#         # разные пороги входа и выхода из сваливания — hysteresis
#         alpha_on = ac.alpha_stall_on
#         alpha_off = ac.alpha_stall_off
#
#         a = abs(self.alpha_eff)
#
#         if a > alpha_on:
#             target_sep = 1.0
#         elif a < alpha_off:
#             target_sep = 0.0
#         else:
#             target_sep = self.sep
#
#         # скорость развития отрыва потока
#         tau_sep = ac.tau_sep_stall if target_sep > self.sep else ac.tau_sep_recover
#         self.sep += dt * (target_sep - self.sep) / tau_sep
#         self.sep = np.clip(self.sep, 0.0, 1.0)
#
#     def CL_static_attached(self, alpha):
#         ac = self.aircraft
#         return ac.CL0 + ac.CL_alpha * alpha
#
#     def CL_post_stall(self, alpha):
#         ac = self.aircraft
#         s = np.sign(alpha)
#         a = abs(alpha)
#
#         # плато после сваливания
#         plateau = ac.CL_plateau
#
#         # возможный повторный рост на больших углах за счёт фюзеляжного/вихревого вклада
#         vortex = ac.CL_vortex_gain * np.sin(2 * a) ** 2
#
#         # спад после критического угла
#         decay = np.exp(-ac.CL_decay * max(0.0, a - ac.alpha_stall_on))
#
#         cl = plateau + vortex
#         cl = min(cl, ac.CL_max * decay + plateau * (1.0 - decay) + vortex)
#
#         return s * cl
#
#     def compute_CL(self, alpha, q_hat, dt, turbulence=0.0):
#         self.update_dynamic_states(alpha, q_hat, dt)
#
#         cl_attached = self.CL_static_attached(self.alpha_eff)
#         cl_attached = np.clip(cl_attached, -self.aircraft.CL_max, self.aircraft.CL_max)
#
#         cl_stalled = self.CL_post_stall(self.alpha_eff)
#
#         # смешивание присоединённого и сорванного режимов
#         CL = (1.0 - self.sep) * cl_attached + self.sep * cl_stalled
#
#         # динамический прирост при быстром тангаже
#         CL += self.aircraft.CL_q * q_hat * (1.0 - 0.5 * self.sep)
#
#         # турбулентные флуктуации после срыва
#         CL += turbulence * self.sep
#
#         return CL