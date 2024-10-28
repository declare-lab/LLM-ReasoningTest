import pandas as pd
from fire import Fire


def math_result(filename: str, level: int = 0, label_column_name="auto_label"):
    """level: 0, 1, 2, 3"""
    print(filename)
    df = pd.read_csv(filename)
    if level == 0:
        df = df[~df["category"].isin(["invalid", "original"])]
        mean = df[label_column_name].mean()
        print(mean)
        return
    if level == 1:
        key = "aspect"
    elif level == 2:
        key = "target"
    elif level == 3:
        key = "dimension"
    elif level == 4:
        key = "category"
    mean = df.groupby(key)[label_column_name].mean()
    print(mean)


def code_result(filename: str, level: int = 0, label_column_name="auto_label"):
    """level: 0, 1, 2, 3"""
    print(filename)
    df = pd.read_csv(filename)
    if df.auto_label.dtype != bool:
        df["auto_label"] = df["auto_label"] == "True"
    if level == 0:
        df = df[~df["category"].isin(["invalid", "original"])]
        mean = df[label_column_name].mean()
        print(mean)
        return
    if level == 1:
        key = "aspect"
    elif level == 2:
        key = "target"
    elif level == 3:
        key = "dimension"
    elif level == 4:
        key = "category"
    mean = df.groupby(key)[label_column_name].mean()
    print(mean)


def autoeval_result(filename: str):
    """level: 0, 1, 2, 3"""
    print(filename)
    df = pd.read_csv(filename)
    df = df[~df["category"].isin(["invalid", "original"])]
    results = []
    for (
        a,
        h,
    ) in zip(df["auto_label"], df["human_label"]):
        a = str(a).lower()
        h = str(h).lower()
        if str(a) == "nan" or str(h) == "nan":
            continue
        results.append(a == h)
    print(sum(results) / len(results))


if __name__ == "__main__":
    Fire()
