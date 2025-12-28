import pandas as pd
from pathlib import Path

def build_ml_data(df):
    agg_dict = {
        'win': 'sum',         
        'PTS': 'mean',           
        'points_against': 'mean',
        'FGM': 'mean',
        'FGA': 'mean',
        'FG3M': 'mean',
        'FG3A': 'mean',
        'FTM': 'mean',
        'FTA': 'mean',
        'REB': 'mean',
        'OREB': 'mean',
        'DREB': 'mean',
        'AST': 'mean',
        'TOV': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'PF': 'mean',
        'point_diff': 'mean'
        }
    team_season = df.groupby(['TEAM_ID', 'season']).agg(agg_dict).reset_index()
    team_season['FG_pct'] = team_season['FGM'] / team_season['FGA']
    team_season['3P_pct'] = team_season['FG3M'] / team_season['FG3A']
    team_season['FT_pct'] = team_season['FTM'] / team_season['FTA']
    return team_season
def save_data(season):
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[2]
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    ML_DATA_DIR = PROJECT_ROOT / "data" / "ml"
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed_file = PROCESSED_DATA_DIR / f"games_{season}.parquet"
    df = pd.read_parquet(processed_file)
    df['season'] = season
    team_season_df = build_ml_data(df)
    
    output_path = ML_DATA_DIR / f"team_season_{season}.parquet"
    team_season_df.to_parquet(output_path)

    return output_path

def process_seasons():
    seasons = ["2010-11", "2011-12", "2012-13", 
               "2013-14","2014-15", "2015-16",
               "2016-17","2017-18","2018-19",
               "2019-20","2020-21","2021-22",
               "2022-23","2023-24","2024-25"]
    for season in seasons:
        save_data(season)
    return None

if __name__ == "__main__":
    process_seasons()
    print(f"Saved all processed data")

    