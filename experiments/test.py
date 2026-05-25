# import pandas as pd
#
# df = pd.read_csv("results/all_runs_ml_ready.csv")
#
# print("=== ОБЩАЯ СТАТИСТИКА ===")
#
# # --- по точкам ---
# point_counts = df["is_stall"].value_counts()
# print("\nПо точкам:")
# print(point_counts)
#
# # --- по прогонам ---
# runs = df.groupby("run_id")["is_stall"].max()
#
# total_runs = len(runs)
# stall_runs = runs.sum()
#
# print("\nПо прогонам:")
# print(f"Всего прогонов: {total_runs}")
# print(f"Со stall: {stall_runs}")
# print(f"Доля: {stall_runs / total_runs:.3f}")
#
#
# # =========================
# # 📊 ПО СЦЕНАРИЯМ
# # =========================
#
# print("\n=== ПО СЦЕНАРИЯМ ===")
#
# # для каждого run берём scenario + был ли stall
# run_info = df.groupby("run_id").agg({
#     "scenario": "first",
#     "is_stall": "max"
# })
#
# scenario_stats = run_info.groupby("scenario").agg(
#     total_runs=("is_stall", "count"),
#     stall_runs=("is_stall", "sum")
# )
#
# scenario_stats["stall_ratio"] = scenario_stats["stall_runs"] / scenario_stats["total_runs"]
#
# print(scenario_stats.sort_values("stall_ratio", ascending=False))
#
#
# # =========================
# # 🔥 ДОПОЛНИТЕЛЬНО
# # =========================
#
# print("\n=== СЦЕНАРИИ БЕЗ STALL ===")
#
# no_stall = scenario_stats[scenario_stats["stall_runs"] == 0]
# print(no_stall.index.tolist())

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
#
# fig, ax = plt.subplots(figsize=(14, 9))
# ax.set_xlim(0, 14)
# ax.set_ylim(0, 9)
# ax.axis('off')
#
# def add_box(ax, x, y, w, h, text, color='lightblue'):
#     box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
#                          edgecolor='black', facecolor=color, linewidth=1.5)
#     ax.add_patch(box)
#     ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)
#
# def add_arrow(ax, x1, y1, x2, y2, style='->', color='black'):
#     ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
#                 arrowprops=dict(arrowstyle=style, color=color, lw=1.5))
#
# # Рисуем модули
# add_box(ax, 1, 7, 2.5, 0.8, 'AircraftModel')
# add_box(ax, 1, 5.5, 2.5, 0.8, 'Atmosphere')
# add_box(ax, 1, 4, 2.5, 0.8, 'TurbulenceModel')
#
# add_box(ax, 4.5, 5.5, 3.5, 1.2, 'AircraftDynamics\n(derivatives)')
#
# add_box(ax, 6, 7.5, 2, 0.8, 'RiskModel')
#
# add_box(ax, 9, 5.5, 3, 1.2, 'RK2Integrator\n(step)')
#
# add_box(ax, 9, 2.5, 2.5, 0.8, 'Visualizer')
#
# add_box(ax, 0.5, 1.5, 3, 1.2, 'run_experiments.py\nsensitivity.py')
# add_box(ax, 10, 1.5, 3, 0.8, 'validation.py')
#
# add_box(ax, 5, 0.5, 2.5, 0.8, 'CSV-файлы')
#
# # Сценарий
# add_box(ax, 0.5, 7.5, 2, 0.8, 'Scenario')
#
# # Стрелки
# # Scenario -> Integrator
# add_arrow(ax, 2.5, 7.9, 9, 6.3)
# # Integrator -> AircraftDynamics (вызов)
# add_arrow(ax, 9, 6.1, 8, 6.1)
# add_arrow(ax, 8, 5.9, 9, 5.9)  # обратная связь
# # AircraftDynamics использует вспомогательные модули
# add_arrow(ax, 3.5, 5.9, 4.5, 5.9)
# add_arrow(ax, 2, 5.5, 4.5, 6.2)
# add_arrow(ax, 2, 4.4, 4.5, 5.8)
# # Integrator -> RiskModel
# add_arrow(ax, 9.5, 6.7, 7, 7.7)
# # RiskModel -> возврат
# add_arrow(ax, 7, 7.9, 9.5, 6.7, style='<-')
# # Integrator -> Visualizer
# add_arrow(ax, 10.5, 5.5, 10.5, 3.3)
#
# # Пакетный запуск -> Integrator, Scenario, AircraftModel
# add_arrow(ax, 3, 2.1, 6, 5.5, color='gray')
# add_arrow(ax, 2, 2.1, 1, 7, color='gray')
# # Пакетный запуск -> CSV
# add_arrow(ax, 4, 1.5, 5, 1)
#
# # validation.py -> AircraftModel, AircraftDynamics
# add_arrow(ax, 11.5, 2.3, 3.5, 7, color='green')
# add_arrow(ax, 11.5, 2.3, 6, 5.5, color='green')
#
# # Легенда
# ax.text(0.5, 9.5, 'Рисунок 3.1 — Архитектура программного комплекса', fontsize=12, weight='bold')
# plt.tight_layout()
# plt.savefig('architecture.png', dpi=200)
# plt.show()

