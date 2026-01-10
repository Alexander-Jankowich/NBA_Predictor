import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from funcs import load_data, evaluate, load_advanced,load_all
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import TimeSeriesSplit

(
    X_train, y_train,
    X_val, y_val, meta_val,
    X_test, y_test, meta_test
) = load_data()


#aX_train, ay_train, aX_val, ay_val, aX_test, ay_test = load_advanced()

#allx_train,ally_train,allx_val,ally_val,allx_test,ally_test = load_all()

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

ALL = [
        'points_against','FTA',
        'FG_pct', '3P_pct', 'FT_pct',
        'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF',
        'FG3A','FGA','PACE','E_OFF_RATING','E_DEF_RATING','REB_PCT', 
        'AST_PCT','TS_PCT','OREB_PCT', 'DREB_PCT'
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

# def train_advanced():
#     model = RandomForestRegressor(
#     n_estimators=257,
#     max_depth=12,
#     criterion='squared_error',
#     min_samples_leaf=2,
#     max_features=0.75,
#     random_state= None
#     )
#     preds = model.fit(X_train, y_train)
#     val_preds = model.predict(X_val)
#     val_mae = mean_absolute_error(y_val, val_preds)
#     val_r2 = r2_score(y_val, val_preds)
#     print("Forest Validation MAE:", val_mae)
#     print("Forest R2:", val_r2)
#     return model

# def train_all():
#     model = RandomForestRegressor(
#     n_estimators=595,
#     max_depth=96,
#     criterion='absolute_error',
#     min_samples_leaf=1,
#     max_features=0.75,
#     random_state= None
#     )
#     preds = model.fit(X_train, y_train)
#     val_preds = model.predict(X_val)
#     val_mae = mean_absolute_error(y_val, val_preds)
#     val_r2 = r2_score(y_val, val_preds)
#     print("Forest Validation MAE:", val_mae)
#     print("Forest R2:", val_r2)
#     return model

def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': randint(300, 1500),
        'max_depth': randint(10, 100),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.5, 0.75],
        'criterion': ['squared_error', 'absolute_error']
    }

    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=300,                  # number of random trials
        scoring='neg_mean_squared_error',
        cv=tscv,                       # cross-validation folds
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
    feature_importance = pd.Series(importances, index=ALL)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    feature_importance.plot(kind='barh')
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def print_wins(model):
    """
    Print predicted and actual wins for all teams, grouped by season.
    
    Parameters:
        model: Trained regression model
    """
    # Get predictions
    preds = model.predict(X_test)
    
    # Create DataFrame with metadata
    pred_df = meta_test.copy()
    pred_df['predicted_wins'] = preds
    pred_df['actual_wins'] = y_test.values
    pred_df['error'] = pred_df['predicted_wins'] - pred_df['actual_wins']
    
    # Sort by season, then predicted wins descending
    pred_df = pred_df.sort_values(['season', 'predicted_wins'], ascending=[True, False])
    
    # Print season by season
    for season in pred_df['season'].unique():
        print(f"\n=== {season} ===")
        season_df = pred_df[pred_df['season'] == season]
        print(season_df[['TEAM_NAME', 'predicted_wins', 'actual_wins', 'error']].reset_index(drop=True))




if __name__ == "__main__":
    model = train_forest()
    #evaluate(model,X_test,y_test)
    #model = train_advanced()
    evaluate(model,X_test,y_test)
    #plot_features(model)
    #model = train_all()
    #evaluate(model,X_test,y_test)
    #plot_features(model)
    print_wins(model)
    #tune_random_forest(X_train,y_train)
    #tune_random_forest(aX_train,ay_train)
    #tune_random_forest(allx_train,ally_train)

