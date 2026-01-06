import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
ML_DATA_DIR = PROJECT_ROOT / "data" / "ml"
files = sorted(ML_DATA_DIR.glob("team_season_*.parquet"))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(['season', 'TEAM_ID']).reset_index(drop=True)

train = df[df['season'] <= '2018-19']
val = df[('2019-20' <= df['season']) & (df['season'] <= '2021-22')]
test = df[df['season'] > '2021-22']

FEATURES = [
    'points_against', 'point_diff','FTA',
    'FG_pct', '3P_pct', 'FT_pct',
    'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF',
    'FG3A','FGA'
]

X_train, y_train = train[FEATURES], train['win']
X_val, y_val = val[FEATURES], val['win']
X_test, y_test = test[FEATURES], test['win']

def train_forest():
    model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
    )
    preds = model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)
    print("Forest Validation MAE:", val_mae)
    print("Forest R2:", val_r2)
    return model

if __name__ == "__main__":
    train_forest()
    print("Done")
