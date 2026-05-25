def add_labels(df):
    df = df.copy()

    # 1. базовое событие сваливания
    df["is_stall"] = (df["mode"] == "STALL").astype(int)

    # 2. предупреждение (ранний сигнал)
    df["is_warning"] = (df["mode"] == "WARNING").astype(int)

    df["stall_event"] = (df["sep"] > 0.7).astype(int)

    return df
