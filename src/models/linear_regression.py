import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from funcs import load_data, evaluate


(
    X_train, y_train,
    X_val, y_val, meta_val,
    X_test, y_test, meta_test
) = load_data()

def train_ridge():
    model = LinearRegression()
    model = Ridge(alpha = 10.0)
    model.fit(X_train,y_train)
    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)

    print("Ridge Validation MAE:", val_mae)
    print("Ridge R2:", val_r2)
    return model

def train_val():
    model = LinearRegression()
    model.fit(X_train,y_train)

    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)

    print("Validation MAE:", val_mae)
    print("R2:", val_r2)
    return model

def test(model):
    pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, pred)
    test_r2 = r2_score(y_val, pred)
    print("Test MAE:", test_mae)
    print("Test R2:", test_r2)

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
    model = train_val()
    evaluate(model,X_test,y_test)
    ridge = train_ridge()
    evaluate(ridge,X_test,y_test)
    print_wins(model)
    print_wins(ridge)


    


