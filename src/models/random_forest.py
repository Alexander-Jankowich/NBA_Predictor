import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from funcs import load_data, evaluate

X_train, y_train, X_val, y_val, X_test, y_test = load_data()

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
    model = train_forest()
    evaluate(model,X_test,y_test)

