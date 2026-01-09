import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from funcs import load_data, evaluate
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test = load_data()

FEATURES = [
    # Scoring
    'PTS', 'points_against', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    
    # Team box-score stats
    'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF',
    
    # Efficiency percentages
    'FG_pct', '3P_pct', 'FT_pct', 
    
    # Pace
    'PACE',
    
    # Rebounding rates
    'OREB_PCT', 'DREB_PCT', 'REB_PCT',
    
    # Assists
    'AST_PCT'
    ]


def train_forest():
    model = RandomForestRegressor(
    n_estimators=400,
    max_depth=9,
    min_samples_leaf=5,
    random_state= None
    )
    preds = model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)
    print("Forest Validation MAE:", val_mae)
    print("Forest R2:", val_r2)
    return model

def plot_features(model):
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=FEATURES)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    feature_importance.plot(kind='barh')
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = train_forest()
    evaluate(model,X_test,y_test)
    plot_features(model)

