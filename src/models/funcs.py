import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
ML_DATA_DIR = PROJECT_ROOT / "data" / "ml"

def load_data():
    files = sorted(ML_DATA_DIR.glob("team_season_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['season', 'TEAM_ID']).reset_index(drop=True)

    train = df[df['season'] <= '2018-19']
    val = df[('2019-20' <= df['season']) & (df['season'] <= '2021-22')]
    test = df[df['season'] > '2021-22']

    FEATURES = [
        'points_against','FTA',
        'FG_pct', '3P_pct', 'FT_pct',
        'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF',
        'FG3A','FGA'
    ]

    X_train, y_train = train[FEATURES], train['win']
    X_val, y_val = val[FEATURES], val['win']
    X_test, y_test = test[FEATURES], test['win']

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    return mae, r2