
import numpy as np

FEATURES = [
    "V", "theta_rad", "gamma_rad", "alpha_rad",
    "q", "vertical_speed",
    "load_factor",
    "throttle",
    "elevator_command"
]

WINDOW = 20
HORIZON = 10


def make_windows(df):
    X, y = [], []

    for run_id, run in df.groupby("run_id"):
        run = run.sort_values("t")

        values = run[FEATURES].values
        labels = run["stall_event"].values

        for i in range(len(run) - WINDOW - HORIZON):

            X.append(values[i:i+WINDOW])

            # цель: будет ли stall в будущем окне
            future_stall = labels[i+WINDOW:i+WINDOW+HORIZON].any()

            y.append(int(future_stall))

    return np.array(X), np.array(y)