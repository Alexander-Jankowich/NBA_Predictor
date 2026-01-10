import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from funcs import load_data, evaluate, load_advanced
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

X_train, y_train, X_val, y_val, X_test, y_test = load_data()


aX_train, ay_train, aX_val, ay_val, aX_test, ay_test = load_advanced()

FEATURES = [
        'points_against','FTA',
        'FG_pct', '3P_pct', 'FT_pct',
        'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF',
        'FG3A','FGA','PACE'
    ]

ADVANCED = [
       'E_OFF_RATING','E_DEF_RATING','PACE','REB_PCT', 'AST_PCT','TS_PCT','FG_pct',
       '3P_pct','FT_pct'
    ]


def train_forest():
    model = RandomForestRegressor(
    criterion='squared_error',
    n_estimators=720,
    max_depth=18,
    max_features='sqrt',
    min_samples_leaf=2,
    random_state= None
    )
    preds = model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)
    print("Forest Validation MAE:", val_mae)
    print("Forest R2:", val_r2)
    return model

def train_advanced():
    model = RandomForestRegressor(
    n_estimators=659,
    max_depth=5,
    criterion='absolute_error',
    min_samples_leaf=2,
    max_features=0.75,
    random_state= None
    )
    preds = model.fit(aX_train, ay_train)
    val_preds = model.predict(aX_val)
    val_mae = mean_absolute_error(ay_val, val_preds)
    val_r2 = r2_score(ay_val, val_preds)
    print("Forest Validation MAE:", val_mae)
    print("Forest R2:", val_r2)
    return model

def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': randint(200, 800),
        'max_depth': randint(4, 25),
        'min_samples_leaf': randint(2, 20),
        'max_features': ['sqrt', 'log2', 0.5, 0.75],
        'criterion': ['squared_error', 'absolute_error']
    }

    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=40,                  # number of random trials
        scoring='neg_mean_absolute_error',
        cv=3,                       # cross-validation folds
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("Best MAE:", -search.best_score_)
    print("Best Params:", search.best_params_)

    return search.best_estimator_

def plot_features(model):
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=ADVANCED)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    feature_importance.plot(kind='barh')
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #model = train_forest()
    #evaluate(model,X_test,y_test)
    model = train_advanced()
    evaluate(model,aX_test,ay_test)
    plot_features(model)
    #tune_random_forest(X_train,y_train)
    #tune_random_forest(aX_train,ay_train)

