import pandas as pd

CATEGORICAL_COLS = ['from', 'to', 'flightType', 'agency']
NUMERICAL_COLS = ['time', 'distance', 'month']
TARGET = 'price'

def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df[TARGET]

    return X, y
