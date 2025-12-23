from nba_api.stats.endpoints import leaguegamelog
import pandas
import os
import time


def fetch_season_log(season):
    """
    Fetches the game logs for a given season and returns them (expected (2460,29))

    Parameters: season (string): Of form "xxxx-xxxx". EX: 2022-2023
    """
    log = leaguegamelog.LeagueGameLog(
        season = season,
        season_type_all_star = "Regular Season"
    ).get_data_frames()[0]
    return log

def save_raw_data(df, season):
    """
    Save the raw league game log DataFrame to data/raw as a parquet file.
    
    Parameters:
        df (pd.DataFrame): DataFrame returned from fetch_season_log
        season (str): Season string, e.g., "2022-23"
    
    Returns:
        str: Path to the saved file
    """
    
