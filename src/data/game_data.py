from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import os
import time
from pathlib import Path


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

def save_raw_data(season):
    """
    Save the raw league game log DataFrame to data/raw as a parquet file.
    
    Parameters:
        season (str): Season string, e.g., "2022-23"
    
    Returns:
        str: Path to the saved file
    """
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[2]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_season_log(season)
    output_path = RAW_DATA_DIR / f"games_{season}.parquet"
    df.to_parquet(output_path)
    return output_path

def extract_seasons():
    seasons = ["2010-11", "2011-12", "2012-13", 
               "2013-14","2014-15", "2015-16",
               "2016-17","2017-18","2018-19",
               "2019-20","2020-21","2021-22",
               "2022-23","2023-24","2024-25"]
    for season in seasons:
        save_raw_data(season)
        time.sleep(1.5)
    return None

if __name__ == "__main__":
    extract_seasons()
    print(f"Saved all raw data")
