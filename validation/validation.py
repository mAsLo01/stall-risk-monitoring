"""
validation.py

Валидационный модуль для проверки базовых скоростей сваливания модели Cessna 172S.

Проверяется:
- соответствие Vstall в конфигурации CLEAN табличному значению POH;
- соответствие Vstall в конфигурации LANDING табличному значению POH;
- рост скорости сваливания при крене 30°, 45°, 60° через формулу:
      Vs_bank = Vs_1g / sqrt(cos(bank_angle))

Важно:
модуль не запускает динамическую симуляцию. Он проверяет только статическую
калибровку модели по опубликованным характеристикам.
"""

import numpy as np

from AircraftModel import CESSNA_172, CLEAN, LANDING
from model.dynamics import AircraftDynamics


def mps_to_kcas(v_mps: float) -> float:
    """
    Перевод м/с в KCAS.

    В рамках данной проверки считаем, что calibrated airspeed используется
    как табличная скорость из POH, без дополнительной модели приборных поправок.
    """
    return v_mps / 0.514444


def kcas_to_mps(v_kcas: float) -> float:
    return v_kcas * 0.514444


def stall_speed_in_bank(vs_1g: float, bank_deg: float) -> float:
    """
    Расчёт скорости сваливания в координированном развороте.

    n = 1 / cos(phi)
    Vs_bank = Vs_1g * sqrt(n) = Vs_1g / sqrt(cos(phi))

    Здесь предполагается установившийся координированный разворот.
    """
    phi = np.radians(bank_deg)
    cos_phi = np.cos(phi)

    if cos_phi <= 0.0:
        raise ValueError("Bank angle must be less than 90 degrees.")

    return vs_1g / np.sqrt(cos_phi)


def percent_error(model_value: float, reference_value: float) -> float:
    return 100.0 * (model_value - reference_value) / reference_value


def validate_configuration(name: str, config, poh_table_kcas: dict[int, float], tolerance_kcas: float = 1.5):
    """
    Проверяет одну конфигурацию самолёта: CLEAN или LANDING.

    name — имя конфигурации для печати.
    config — AeroConfig.
    poh_table_kcas — табличные значения POH вида:
        {0: 53, 30: 57, 45: 63, 60: 75}
    tolerance_kcas — допустимое абсолютное отклонение.
    """

    aircraft = CESSNA_172
    dynamics = AircraftDynamics(aircraft, config)

    rho0 = 1.225

    vs_1g_mps = dynamics.compute_v_stall(rho0)
    vs_1g_kcas = mps_to_kcas(vs_1g_mps)

    print()
    print(f"=== Validation: {name} ===")
    print(f"Config CL_max = {config.CL_max:.3f}")
    print(f"Model Vs_1g = {vs_1g_kcas:.2f} KCAS")
    print()

    print(
        f"{'Bank':>6} | "
        f"{'POH, KCAS':>9} | "
        f"{'Model, KCAS':>11} | "
        f"{'Abs err':>8} | "
        f"{'Err, %':>7} | "
        f"{'Status':>8}"
    )
    print("-" * 64)

    all_passed = True

    for bank_deg, poh_vs_kcas in poh_table_kcas.items():
        if bank_deg == 0:
            model_vs_kcas = vs_1g_kcas
        else:
            model_vs_mps = stall_speed_in_bank(vs_1g_mps, bank_deg)
            model_vs_kcas = mps_to_kcas(model_vs_mps)

        abs_err = model_vs_kcas - poh_vs_kcas
        err_pct = percent_error(model_vs_kcas, poh_vs_kcas)

        passed = abs(abs_err) <= tolerance_kcas
        all_passed = all_passed and passed

        status = "PASS" if passed else "CHECK"

        print(
            f"{bank_deg:>6.0f} | "
            f"{poh_vs_kcas:>9.2f} | "
            f"{model_vs_kcas:>11.2f} | "
            f"{abs_err:>8.2f} | "
            f"{err_pct:>7.2f} | "
            f"{status:>8}"
        )

    return all_passed


def run_validation():
    """
    Главная функция валидации.

    Табличные значения взяты для Cessna 172S, масса 2550 lb,
    Power Off, KCAS, углы крена 0°, 30°, 45°, 60°.
    """

    poh_clean_kcas = {
        0: 53.0,
        30: 57.0,
        45: 63.0,
        60: 75.0,
    }

    poh_landing_kcas = {
        0: 48.0,
        30: 52.0,
        45: 57.0,
        60: 68.0,
    }

    clean_ok = validate_configuration(
        name="CLEAN / flaps UP",
        config=CLEAN,
        poh_table_kcas=poh_clean_kcas,
    )

    landing_ok = validate_configuration(
        name="LANDING / flaps FULL",
        config=LANDING,
        poh_table_kcas=poh_landing_kcas,
    )

    print()
    print("=== Summary ===")

    if clean_ok and landing_ok:
        print("All basic stall-speed validation checks passed.")
    else:
        print("Some checks need attention. Check CL_max calibration, mass, rho, or POH reference values.")


if __name__ == "__main__":
    run_validation()