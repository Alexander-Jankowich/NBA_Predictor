from pathlib import Path
import pandas as pd
RAW_DATA_DIR = Path("data/raw")
raw_files = sorted(RAW_DATA_DIR.glob("games_*.parquet"))

def build_team_data(raw_df):
    """
    Convert a raw season DataFrame into a per-team, per-game DataFrame
    with explicit win/loss, points, home/away, and opponent info.
    
    Input:
        raw_df : pd.DataFrame
            Raw game logs from nba_api for one season
    Output:
        team_game_df : pd.DataFrame
            Structured table: one row per team per game
    """ 
    df = raw_df.copy()

    df['win'] = df['WL'].map({'W':1, 'L':0})
    df['is_home'] = df['MATCHUP'].str.contains('vs').astype(int)
    df['opponent_abbr'] = df['MATCHUP'].str.split(' ').str[-1]

    opp_stats = df[['GAME_ID','TEAM_ID','PTS']].rename(
        columns={'TEAM_ID':'opponent_team_id','PTS':'points_against'}
    )
    df = df.merge(opp_stats, on='GAME_ID')
    df = df[df['TEAM_ID'] != df['opponent_team_id']]
    df['point_diff'] = df['PTS'] - df['points_against']

    stats_cols = ['FGM', 'FGA','FG3M','FG3A','FTM', 'FTA', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF']
    stats_cols = [col for col in stats_cols if col in df.columns]
    
    df = df[['TEAM_ID', 'GAME_ID', 'opponent_abbr', 'is_home', 'win', 'PTS', 'points_against', 'point_diff'] + stats_cols]    

    assert df['GAME_ID'].nunique() * 2 == len(df)
    assert df['win'].sum() <= len(df) 
    return df

if __name__ == "__main__":
    PROCESSED_DATA_DIR = Path("data/processed")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in raw_files:
        print(f"Processing {file_path.name}...")
        raw_df = pd.read_parquet(file_path)         
        team_df = build_team_data(raw_df)                
        output_path = PROCESSED_DATA_DIR / file_path.name  
        team_df.to_parquet(output_path)                    

