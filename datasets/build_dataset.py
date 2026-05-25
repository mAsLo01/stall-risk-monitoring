import pandas as pd
from .labels import add_labels

def load_dataset(path):
    df = pd.read_csv(path)

    df = add_labels(df)

    df = df.sort_values(["run_id", "t"])

    return df