import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.features.engineering import build_features

TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"


def make_dataset(df: pd.DataFrame) -> dict:
    """Build features, split by date, and scale using training data only."""
    features = build_features(df).dropna()  # drop warm-up rows

    train = features.loc[:TRAIN_END]
    test = features.loc[TEST_START:]

    scaler = StandardScaler().fit(train)  # learns mean/std from TRAIN only

    train_scaled = pd.DataFrame(
        scaler.transform(train), index=train.index, columns=train.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test), index=test.index, columns=test.columns
    )

    return {
        "train": train,
        "test": test,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
    }