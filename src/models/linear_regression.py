import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
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
    'PTS', 'points_against', 'point_diff',
    'FG_pct', '3P_pct', 'FT_pct',
    'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF',
    'FG3A'
]

X_train, y_train = train[FEATURES], train['win']
X_val, y_val = val[FEATURES], val['win']
x_test, y_test = test[FEATURES], test['win']

def train_test():
    model = LinearRegression()
    model.fit(X_train,y_train)

    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)

    print("Validation MAE:", val_mae)
    print("R2:", val_r2)

if __name__ == "__main__":
    train_test()
    print(f"Saved all processed data")

    


