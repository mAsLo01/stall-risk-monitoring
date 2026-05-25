"""
Глобальные численные параметры симуляции.
Физические параметры перенесены в AircraftModel и Atmosphere.
"""

def kts_to_mps(kts):
    return kts * 0.514444


STALL_WARNING_MARGIN_KTS = 7.0
STALL_WARNING_MARGIN_MPS = kts_to_mps(STALL_WARNING_MARGIN_KTS)

dt = 0.1 # Шаг моделирования (сек).


risk_decay_time = 2.0 # Характерное время "забывания" риска (сек). Используется для вычисления λ



V_cruise_ref = 50.0 # Референсная крейсерская скорость (м/с), используется для нормировки энергии.


min_v = 5.0

