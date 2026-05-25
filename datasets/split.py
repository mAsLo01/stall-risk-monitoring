import numpy as np
def split_by_run(df, train_ratio=0.8):
    runs = df["run_id"].unique()
    np.random.shuffle(runs)

    cut = int(len(runs) * train_ratio)

    train_runs = runs[:cut]
    test_runs = runs[cut:]

    return (
        df[df["run_id"].isin(train_runs)],
        df[df["run_id"].isin(test_runs)]
    )