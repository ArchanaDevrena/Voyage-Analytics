import pandas as pd

CATEGORICAL_COLS = ['from', 'to', 'flightType', 'agency']
NUMERICAL_COLS = ['time', 'distance', 'month']
TARGET = 'price'

def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    
    # Ensure 'date' exists, otherwise fallback to current month
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
    else:
        df['month'] = 1  # default month if missing

    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df[TARGET] if TARGET in df.columns else None

    return X, y
