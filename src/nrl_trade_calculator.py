import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from itertools import combinations
from datetime import datetime
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

@dataclass
class Player:
    name: str
    price: int
    position: str
    points: int
    total_base: int
    base_premium: int
    consecutive_good_weeks: int
    age: int


def load_data() -> pd.DataFrame:
    """
    Load data from PostgreSQL database and rename columns to match expected names.
    
    Returns:
    pd.DataFrame: DataFrame with standardized column names
    """
    # Read database connection parameters from environment
    load_dotenv()
    db_params = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_DATABASE'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT')
    }
    
    # Create connection string
    conn_str = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    engine = create_engine(conn_str)
    
    try:
        # First, let's see what columns we actually have in the database
        with engine.connect() as connection:
            # Get column names from the table
            query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'player_stats';
            """
            db_columns = pd.read_sql(query, connection)
            
            # Now fetch the actual data
            query = "SELECT * FROM player_stats;"
            df = pd.read_sql(query, connection)
        
        # Ensure required columns exist
        required_columns = ['Round', 'Team', 'POS1', 'Player', 'Price', 'Diff', 'Projection']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert Round to integer
        df['Round'] = df['Round'].astype(int)
        
        # Handle POS2 if it exists
        if 'POS2' not in df.columns:
            df['POS2'] = None
            
        return df
        
    except Exception as e:
        print(f"Error loading data from database: {str(e)}")
        raise

def get_rounds_data(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Split consolidated data into list of DataFrames by round.
    """
    rounds = sorted(df['Round'].unique())
    return [df[df['Round'] == round_num].copy() for round_num in rounds]

def check_consistent_performance(player_name: str, consolidated_data: pd.DataFrame, min_base_premium: int = 5, required_consecutive_weeks: int = 2, player_histories: Dict[str, pd.DataFrame] = None) -> int:
    """
    Check how many consecutive weeks a player has maintained good performance.
    """
    if player_histories is not None and player_name in player_histories:
        player_data = player_histories[player_name]
    else:
        player_data = consolidated_data[consolidated_data['Player'] == player_name].sort_values('Round')
    
    if player_data.empty:
        return 0
        
    consecutive_weeks = 0
    current_streak = 0
    
    for _, row in player_data.iterrows():
        if row['Base exceeds price premium'] >= min_base_premium:
            current_streak += 1
        else:
            current_streak = 0
            
        consecutive_weeks = max(consecutive_weeks, current_streak)
    
    return consecutive_weeks

def check_rule_condition(
    player_data: pd.Series,
    consolidated_data: pd.DataFrame,
    base_premium_threshold: int,
    weeks_threshold: int,
    position_requirement: str = None,
    max_age: int = None,
    player_histories: Dict[str, pd.DataFrame] = None
) -> bool:
    """
    Check if a player meets the specified rule conditions.
    Exclude Mid/EDG players who are 29 years or older.
    """
    # Exclude Mid/EDG players who are 29 years or older
    if player_data['POS1'] in ['MID', 'EDG'] and player_data['Age'] >= 29:
        return False
    
    # Check base premium threshold for current week
    meets_bpre = player_data['Base exceeds price premium'] >= base_premium_threshold
    
    # Check consecutive weeks requirement
    player_name = player_data['Player']
    
    if player_histories is not None and player_name in player_histories:
        player_history = player_histories[player_name].sort_values('Round')
    else:
        player_history = consolidated_data[consolidated_data['Player'] == player_name].sort_values('Round')
    
    # Check if the player has met the threshold for the required consecutive weeks
    current_streak = 0
    for _, row in player_history.iterrows():
        if row['Base exceeds price premium'] >= base_premium_threshold:
            current_streak += 1
        else:
            current_streak = 0
        
        if current_streak >= weeks_threshold:
            break
    
    meets_weeks = current_streak >= weeks_threshold
    
    if position_requirement:
        positions = position_requirement.split('|')
        meets_position = player_data['POS1'] in positions
    else:
        meets_position = True
        
    if max_age is not None:
        meets_age = player_data['Age'] <= max_age
    else:
        meets_age = True
        
    return meets_bpre and meets_weeks and meets_position and meets_age

def assign_priority_level(player_data: pd.Series, consolidated_data: pd.DataFrame, player_histories: Dict[str, pd.DataFrame] = None) -> int:
    """
    Assign priority level based on the updated rules.
    """
    # Rule 1: BPRE >= 14 for last 3 weeks
    if check_rule_condition(player_data, consolidated_data, 14, 3, player_histories=player_histories):
        return 1
        
    # Rule 2: BPRE >= 21 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 21, 2, player_histories=player_histories):
        return 2
        
    # Rule 3: 2 week Average BPRE >= 26
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 26:
        return 3
        
    # Rule 4: BPRE >= 12 for last 3 weeks
    if check_rule_condition(player_data, consolidated_data, 12, 3, player_histories=player_histories):
        return 4
        
    # Rule 5: BPRE >= 19 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 19, 2, player_histories=player_histories):
        return 5
        
    # Rule 6: 2 week Average BPRE >= 24
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 24:
        return 6
        
    # Rule 7: BPRE >= 10 for last 3 weeks
    if check_rule_condition(player_data, consolidated_data, 10, 3, player_histories=player_histories):
        return 7
        
    # Rule 8: BPRE >= 17 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 17, 2, player_histories=player_histories):
        return 8
        
    # Rule 9: 2 week Average BPRE >= 22
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 22:
        return 9
        
    # Rule 10: BPRE >= 8 for last 3 weeks
    if check_rule_condition(player_data, consolidated_data, 8, 3, player_histories=player_histories):
        return 10
        
    # Rule 11: BPRE >= 15 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 15, 2, player_histories=player_histories):
        return 11
        
    # Rule 12: 2 week Average BPRE >= 20
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 20:
        return 12
        
    # Rule 13: BPRE >= 6 for last 3 weeks
    if check_rule_condition(player_data, consolidated_data, 6, 3, player_histories=player_histories):
        return 13
        
    # Rule 14: BPRE >= 13 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 13, 2, player_histories=player_histories):
        return 14
        
    # Rule 15: 2 week Average BPRE >= 18
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 18:
        return 15
        
    # Rule 16: BPRE >= 10 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 10, 2, player_histories=player_histories):
        return 16
        
    # Rule 17: 2 week Average BPRE >= 15
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 15:
        return 17
        
    # Rule 18: BPRE >= 8 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 8, 2, player_histories=player_histories):
        return 18
        
    # Rule 19: 2 week Average BPRE >= 13
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 13:
        return 19
        
    # Rule 20: BPRE >= 6 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 6, 2, player_histories=player_histories):
        return 20
        
    # Rule 21: 2 week Average BPRE >= 11
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 11:
        return 21
        
    # Rule 22: BPRE >= 2 for last 3 weeks
    if check_rule_condition(player_data, consolidated_data, 2, 3, player_histories=player_histories):
        return 22
        
    # Rule 23: BPRE >= 4 for last 2 weeks
    if check_rule_condition(player_data, consolidated_data, 4, 2, player_histories=player_histories):
        return 23
        
    # Rule 24: 2 week Average BPRE >= 9
    if calculate_average_bpre(player_data['Player'], consolidated_data, 2, player_histories=player_histories) >= 9:
        return 24
        
    # Default - lowest priority
    return 25

def calculate_average_bpre(player_name: str, consolidated_data: pd.DataFrame, lookback_weeks: int = 3, player_histories: Dict[str, pd.DataFrame] = None) -> float:
    """
    Calculate average BPRE for a player over their recent weeks.
    Only calculate the average if the player has played in at least `lookback_weeks` rounds.
    Exclude Mid/EDG players who are 29 years or older.
    """
    if player_histories is not None and player_name in player_histories:
        player_data = player_histories[player_name].sort_values('Round', ascending=False)
    else:
        player_data = consolidated_data[consolidated_data['Player'] == player_name].sort_values('Round', ascending=False)
    
    # Exclude Mid/EDG players who are 29 years or older
    if not player_data.empty:
        if player_data.iloc[0]['POS1'] in ['MID', 'EDG'] and player_data.iloc[0]['Age'] >= 29:
            return 0.0  # Exclude these players
    
    recent_data = player_data.head(lookback_weeks)
    
    # Only calculate the average if the player has played in at least `lookback_weeks` rounds
    if len(recent_data) < lookback_weeks:
        return 0.0  # Return 0 if the player hasn't played enough rounds
    
    return int(recent_data['Base exceeds price premium'].mean())

def calculate_average_base(player_name: str, consolidated_data: pd.DataFrame, lookback_weeks: int = 3, min_games: int = 2, player_histories: Dict[str, pd.DataFrame] = None) -> float:
    """
    Calculate average Total base for a player over their recent weeks.
    Only calculate the average if the player has played in at least `min_games` rounds.
    
    Parameters:
    player_name (str): Name of the player
    consolidated_data (pd.DataFrame): DataFrame containing all player data
    lookback_weeks (int): Number of weeks to look back for calculating average
    min_games (int): Minimum number of games required to calculate average
    
    Returns:
    float: Average base value, or 0.0 if minimum games requirement not met
    """
    if player_histories is not None and player_name in player_histories:
        player_data = player_histories[player_name].sort_values('Round', ascending=False)
    else:
        player_data = consolidated_data[consolidated_data['Player'] == player_name].sort_values('Round', ascending=False)
    recent_data = player_data.head(lookback_weeks)
    
    # Check if player has played minimum required games
    if len(recent_data) >= min_games:
        return int(recent_data['Total base'].mean())
    return 0.0

def print_players_by_rule_level(available_players: pd.DataFrame, consolidated_data: pd.DataFrame, maximize_base: bool = False, player_histories: Dict[str, pd.DataFrame] = None) -> None:
    """
    Print players that satisfy each rule level, with their relevant stats.
    """
    print("\n=== Players Satisfying Each Rule Level ===\n")
    rule_descriptions = {
        1: "BPRE >= 14 for last 3 weeks",
        2: "BPRE >= 21 for last 2 weeks",
        3: "2 week Average BPRE >= 26",
        4: "BPRE >= 12 for last 3 weeks",
        5: "BPRE >= 19 for last 2 weeks",
        6: "2 week Average BPRE >= 24",
        7: "BPRE >= 10 for last 3 weeks",
        8: "BPRE >= 17 for last 2 weeks",
        9: "2 week Average BPRE >= 22",
        10: "BPRE >= 8 for last 3 weeks",
        11: "BPRE >= 15 for last 2 weeks",
        12: "2 week Average BPRE >= 20",
        13: "BPRE >= 6 for last 3 weeks",
        14: "BPRE >= 13 for last 2 weeks",
        15: "2 week Average BPRE >= 18",
        16: "BPRE >= 10 for last 2 weeks",
        17: "2 week Average BPRE >= 15",
        18: "BPRE >= 8 for last 2 weeks",
        19: "2 week Average BPRE >= 13",
        20: "BPRE >= 6 for last 2 weeks",
        21: "2 week Average BPRE >= 11",
        22: "BPRE >= 2 for last 3 weeks",
        23: "BPRE >= 4 for last 2 weeks",
        24: "2 week Average BPRE >= 9",
        25: "No rules satisfied"
    }

    for level in range(1, 25):  # Exclude rule 25
        level_players = available_players[available_players['priority_level'] == level]
        
        if not level_players.empty:
            print(f"\nRule Level {level}: {rule_descriptions[level]}")
            print("-" * 80)
            
            # Calculate average BPRE for each player and add it to the DataFrame
            level_players = level_players.copy()
            level_players['avg_bpre'] = level_players['Player'].apply(
                lambda x: calculate_average_bpre(x, consolidated_data, player_histories=player_histories)
            )
            
            # Sort players by average BPRE within the rule level
            level_players_sorted = level_players.sort_values(
                by=['avg_bpre', 'Base exceeds price premium'],
                ascending=[False, False]
            )
            
            for _, player in level_players_sorted.iterrows():
                # Get the player's BPRE for each round in the last 3 rounds
                player_data = consolidated_data[consolidated_data['Player'] == player['Player']].sort_values('Round')
                bpre_by_round = player_data[['Round', 'Base exceeds price premium']].dropna()
                bpre_values = ", ".join([f"round {int(row['Round'])} BPRE: {int(row['Base exceeds price premium'])}" for _, row in bpre_by_round.iterrows()])
                
                print(
                    f"Player: {player['Player']:<20} "
                    f"Team: {player['Team']:<4} "  # Added Team to output
                    f"Position: {player['POS1']:<5} "
                    f"Age: {player['Age']:<3} "
                    f"Current BPRE: {int(player['Base exceeds price premium']):>5} "
                    f"Avg BPRE: {int(player['avg_bpre']):>5} "
                    f"Base: {int(player['Total base']):>5} "
                    f"Price: ${player['Price']:,}"
                )
                print(f"BPRE by Round: {bpre_values}")

def generate_trade_options(
    available_players: pd.DataFrame,
    salary_freed: float,
    maximize_base: bool = False,
    hybrid_approach: bool = False,
    max_options: int = 10,
    trade_type: str = 'likeForLike',
    traded_out_positions: List[str] = None,
    num_players_needed: int = 2
) -> List[Dict]:
    """
    Generate trade combinations based on selected optimization strategy while ensuring
    position requirements are met for both like-for-like and positional swap trades.
    
    Uses Diff for value maximization and Projection for base maximization.
    """
    valid_combinations = []
    used_players = set()
    
    # Make a copy to avoid modifying the original DataFrame
    players_df = available_players.copy()
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Price', 'Diff', 'Projection']
    for col in numeric_columns:
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
    
    # Convert DataFrame to list of dictionaries for easier manipulation
    players = players_df.to_dict('records')
    
    # Create a position mapping for each player
    position_mapping = {}
    for player in players:
        positions = [player['POS1']]
        if pd.notna(player.get('POS2')):
            positions.append(player['POS2'])
        position_mapping[player['Player']] = positions
    
    # Function to check if a player has at least one position from the required positions
    def has_valid_position(player, valid_positions):
        player_positions = position_mapping[player['Player']]
        return any(pos in valid_positions for pos in player_positions)
    
    # Function to check if a player combination works for like-for-like
    def is_valid_like_for_like_combo(player_combo):
        if not traded_out_positions or trade_type != 'likeForLike':
            return True
            
        # First check: each player must have at least one valid position
        if not all(has_valid_position(player, traded_out_positions) for player in player_combo):
            return False
            
        # Second check: all required positions must be covered
        positions_covered = set()
        for player in player_combo:
            for pos in position_mapping[player['Player']]:
                if pos in traded_out_positions:
                    positions_covered.add(pos)
        
        return set(traded_out_positions) == positions_covered
    
    if maximize_base:
        # Sort players by Projection (highest first)
        players.sort(key=lambda x: x['Projection'], reverse=True)
        
        if num_players_needed == 1:
            for player in players:
                if player['Player'] in used_players:
                    continue
                
                if player['Price'] <= salary_freed and is_valid_like_for_like_combo([player]):
                    combo = create_combination([player], player['Price'], salary_freed)
                    valid_combinations.append(combo)
                    used_players.add(player['Player'])
                    if len(valid_combinations) >= max_options:
                        break
        else:
            # For 2+ player trades, find combinations with highest total Projection
            for i in range(len(players)):
                if players[i]['Player'] in used_players:
                    continue
                    
                first_player = players[i]
                
                # Find a valid second player
                for j in range(len(players)):
                    if j == i or players[j]['Player'] in used_players:
                        continue
                        
                    second_player = players[j]
                    
                    # Check if the combination is valid for like-for-like
                    if not is_valid_like_for_like_combo([first_player, second_player]):
                        continue
                    
                    total_price = first_player['Price'] + second_player['Price']
                    if total_price <= salary_freed:
                        combo = create_combination([first_player, second_player], total_price, salary_freed)
                        valid_combinations.append(combo)
                        used_players.add(first_player['Player'])
                        used_players.add(second_player['Player'])
                        break  # Found a valid second player, move to next first player
                
                if len(valid_combinations) >= max_options:
                    break
                    
    elif hybrid_approach:
        # Create two sorted lists
        value_players = sorted(players, key=lambda x: x['Diff'], reverse=True)
        projection_players = sorted(players, key=lambda x: x['Projection'], reverse=True)
        
        if num_players_needed == 1:
            # For single player trades, prioritize value (Diff)
            for player in value_players:
                if player['Player'] in used_players:
                    continue
                
                if player['Price'] <= salary_freed and is_valid_like_for_like_combo([player]):
                    combo = create_combination([player], player['Price'], salary_freed)
                    valid_combinations.append(combo)
                    used_players.add(player['Player'])
                    if len(valid_combinations) >= max_options:
                        break
        else:
            # For 2+ player trades, pair a value player with a projection player
            for value_player in value_players:
                if value_player['Player'] in used_players or value_player['Price'] > salary_freed:
                    continue
                    
                remaining_salary = salary_freed - value_player['Price']
                
                # Find a valid projection player
                found_match = False
                for projection_player in projection_players:
                    if (projection_player['Player'] not in used_players and 
                        projection_player['Player'] != value_player['Player'] and 
                        projection_player['Price'] <= remaining_salary):
                        
                        # Check if the combination is valid for like-for-like
                        if not is_valid_like_for_like_combo([value_player, projection_player]):
                            continue
                        
                        combo = create_combination(
                            [value_player, projection_player],
                            value_player['Price'] + projection_player['Price'],
                            salary_freed
                        )
                        valid_combinations.append(combo)
                        used_players.add(value_player['Player'])
                        used_players.add(projection_player['Player'])
                        found_match = True
                        break
                
                if found_match and len(valid_combinations) >= max_options:
                    break
                    
    else:  # maximize_value - use Diff
        # Sort players by Diff (highest first)
        players.sort(key=lambda x: x['Diff'], reverse=True)
        
        if num_players_needed == 1:
            for player in players:
                if player['Player'] in used_players:
                    continue
                
                if player['Price'] <= salary_freed and is_valid_like_for_like_combo([player]):
                    combo = create_combination([player], player['Price'], salary_freed)
                    valid_combinations.append(combo)
                    used_players.add(player['Player'])
                    if len(valid_combinations) >= max_options:
                        break
        else:
            # For 2+ player trades, find combinations with highest total Diff
            for i in range(len(players)):
                if players[i]['Player'] in used_players:
                    continue
                    
                first_player = players[i]
                
                # Find a valid second player
                found_match = False
                for j in range(len(players)):
                    if j == i or players[j]['Player'] in used_players:
                        continue
                        
                    second_player = players[j]
                    
                    # Check if the combination is valid for like-for-like
                    if not is_valid_like_for_like_combo([first_player, second_player]):
                        continue
                    
                    total_price = first_player['Price'] + second_player['Price']
                    if total_price <= salary_freed:
                        combo = create_combination([first_player, second_player], total_price, salary_freed)
                        valid_combinations.append(combo)
                        used_players.add(first_player['Player'])
                        used_players.add(second_player['Player'])
                        found_match = True
                        break  # Found a valid second player, move to next first player
                
                if found_match and len(valid_combinations) >= max_options:
                    break
    
    return valid_combinations[:max_options]

def create_combination(players, total_price, salary_freed):
    """Helper function to create a trade combination dictionary"""
    return {
        'players': [create_player_dict(player) for player in players],
        'total_price': total_price,
        'total_projection': sum(player.get('Projection', 0) for player in players),
        'total_diff': sum(player.get('Diff', 0) for player in players),
        'salary_remaining': salary_freed - total_price
    }

def create_player_dict(player):
    """Helper function to create consistent player dictionary"""
    return {
        'name': player['Player'],
        'team': player['Team'],
        'position': player['POS1'],
        'secondary_position': player.get('POS2'),
        'price': player['Price'],
        'projection': player.get('Projection', 0),
        'diff': player.get('Diff', 0)
    }

def get_traded_out_positions(traded_out_players: List[str], consolidated_data: pd.DataFrame) -> List[str]:
    """
    Get the positions of the players being traded out.
    
    Parameters:
    traded_out_players (List[str]): List of player names being traded out
    consolidated_data (pd.DataFrame): Full dataset containing player information
    
    Returns:
    List[str]: List of positions corresponding to traded out players
    """
    positions = []
    for player in traded_out_players:
        player_data = consolidated_data[consolidated_data['Player'] == player].sort_values('Round', ascending=False)
        if not player_data.empty:
            positions.append(player_data.iloc[0]['POS1'])
    return positions

def get_locked_out_players(simulate_datetime: str, consolidated_data: pd.DataFrame) -> set:
    """
    Get the set of players who are locked out based on the simulated date/time.
    """
    if not simulate_datetime:
        return set()
    
    # Parse the simulated date/time
    simulate_dt = datetime.strptime(simulate_datetime, '%Y-%m-%dT%H:%M')
    
    # Define the fixtures
    fixtures = [
        ("2025-03-02 11:00", ["CAN", "WAR"]),
        ("2025-03-02 15:30", ["PEN", "CRO"]),
        ("2025-03-06 20:00", ["SYD", "BRI"]),
        ("2025-03-07 18:00", ["WST", "NEW"]),
        ("2025-03-07 20:05", ["DOL", "SOU"]),
        ("2025-03-08 17:30", ["SGI", "CBY"]),
        ("2025-03-08 19:35", ["MAN", "NQL"]),
        ("2025-03-09 16:05", ["MEL", "SOU"]),
    ]
    
    locked_out_teams = set()
    for fixture_time, teams in fixtures:
        fixture_dt = datetime.strptime(fixture_time, '%Y-%m-%d %H:%M')
        if fixture_dt <= simulate_dt:
            locked_out_teams.update(teams)
    
    # Use the Team column from the main data instead of teamlists.csv
    locked_out_players = set()
    latest_round_data = consolidated_data.sort_values('Round').groupby('Player').last().reset_index()
    
    for team in locked_out_teams:
        team_players = latest_round_data[latest_round_data['Team'] == team]['Player'].tolist()
        locked_out_players.update(team_players)
    
    return locked_out_players

def is_player_locked(player_name: str, consolidated_data: pd.DataFrame, simulate_datetime: str) -> bool:
    """
    Check if a player is locked based on the simulated date/time.
    """
    if not simulate_datetime:
        return False
    
    # Parse the simulated date/time
    simulate_dt = datetime.strptime(simulate_datetime, '%Y-%m-%dT%H:%M')
    
    # Define the fixtures
    fixtures = [
        ("2025-03-02 11:00", ["CAN", "WAR"]),
        ("2025-03-02 15:30", ["PEN", "CRO"]),
        ("2025-03-06 20:00", ["SYD", "BRI"]),
        ("2025-03-07 18:00", ["WST", "NEW"]),
        ("2025-03-07 20:05", ["DOL", "SOU"]),
        ("2025-03-08 17:30", ["SGI", "CBY"]),
        ("2025-03-08 19:35", ["MAN", "NQL"]),
        ("2025-03-09 16:05", ["MEL", "SOU"]),
    ]
    
    locked_out_teams = set()
    for fixture_time, teams in fixtures:
        fixture_dt = datetime.strptime(fixture_time, '%Y-%m-%d %H:%M')
        if fixture_dt <= simulate_dt:
            locked_out_teams.update(teams)
    
    # Use the Team column from the main data instead of teamlists.csv
    latest_round_data = consolidated_data.sort_values('Round').groupby('Player').last().reset_index()
    player_team = latest_round_data[latest_round_data['Player'] == player_name]['Team'].values[0]
    
    return player_team in locked_out_teams

def calculate_trade_options(
    consolidated_data: pd.DataFrame,
    traded_out_players: List[str],
    maximize_base: bool = False,
    hybrid_approach: bool = False,
    max_options: int = 10,
    allowed_positions: List[str] = None,
    trade_type: str = 'likeForLike',
    min_games: int = 2,
    team_list: List[str] = None,
    simulate_datetime: str = None,
    apply_lockout: bool = False
) -> List[Dict]:
    # Get locked out players if lockout restriction is applied
    locked_out_players = set()
    if apply_lockout:
        locked_out_players = get_locked_out_players(simulate_datetime, consolidated_data)
    
    # Get positions based on trade type
    if trade_type == 'likeForLike':
        traded_out_positions = get_traded_out_positions(traded_out_players, consolidated_data)
        positions_to_use = None
    elif trade_type == 'positionalSwap':
        traded_out_positions = allowed_positions  # Use selected positions for positional swap
        positions_to_use = allowed_positions
    else:
        traded_out_positions = None
        positions_to_use = None
    
    latest_round = consolidated_data['Round'].max()
    
    # Get number of players needed based on traded out players
    num_players_needed = len(traded_out_players)
    
    # Calculate total salary freed up from traded out players
    salary_freed = 0
    for player in traded_out_players:
        player_data = consolidated_data[consolidated_data['Player'] == player].sort_values('Round', ascending=False)
        if not player_data.empty:
            salary_freed += player_data.iloc[0]['Price']
        else:
            print(f"Warning: Could not find price data for {player}")
    
    print(f"Total salary freed up: ${salary_freed:,}")
    
    # Get all players from the latest round
    latest_round_data = consolidated_data[consolidated_data['Round'] == latest_round]
    available_players = latest_round_data[~latest_round_data['Player'].isin(traded_out_players)]
    
    # Apply team list restriction first if enabled
    if team_list:
        available_players = available_players[available_players['Player'].isin(team_list)]
        if available_players.empty:
            print("Warning: No players available after applying team list restriction")
            return []
    
    # Apply lockout restriction if enabled
    if apply_lockout:
        available_players = available_players[~available_players['Player'].isin(locked_out_players)]
        if available_players.empty:
            print("Warning: No players available after applying lockout restriction")
            return []
    
    # Filter players by allowed positions if specified
    if positions_to_use:
        if trade_type == 'positionalSwap':
            # Check both POS1 and POS2 for allowed positions
            mask = (
                available_players['POS1'].isin(positions_to_use) |
                available_players['POS2'].fillna('').isin(positions_to_use)
            )
            available_players = available_players[mask]
        else:
            # Like-for-like uses only POS1
            available_players = available_players[available_players['POS1'].isin(positions_to_use)]
        if available_players.empty:
            print("Warning: No players available with selected positions")
            return []
    
    # Generate trade options based on the selected strategy
    options = generate_trade_options(
        available_players,
        salary_freed,
        maximize_base,
        hybrid_approach,
        max_options,
        trade_type,
        traded_out_positions,
        num_players_needed
    )
    
    return options[:max_options]

def simulate_rule_levels(consolidated_data: pd.DataFrame, rounds: List[int]) -> None:
    player_name = consolidated_data['Player'].unique()[0]  # Assuming the first player in the data

    # Rule descriptions
    rule_descriptions = {
        1: "BPRE >= 14 for last 3 weeks",
        2: "BPRE >= 21 for last 2 weeks",
        3: "2 week Average BPRE >= 26",
        4: "BPRE >= 12 for last 3 weeks",
        5: "BPRE >= 19 for last 2 weeks",
        6: "2 week Average BPRE >= 24",
        7: "BPRE >= 10 for last 3 weeks",
        8: "BPRE >= 17 for last 2 weeks",
        9: "2 week Average BPRE >= 22",
        10: "BPRE >= 8 for last 3 weeks",
        11: "BPRE >= 15 for last 2 weeks",
        12: "2 week Average BPRE >= 20",
        13: "BPRE >= 6 for last 3 weeks",
        14: "BPRE >= 13 for last 2 weeks",
        15: "2 week Average BPRE >= 18",
        16: "BPRE >= 10 for last 2 weeks",
        17: "2 week Average BPRE >= 15",
        18: "BPRE >= 8 for last 2 weeks",
        19: "2 week Average BPRE >= 13",
        20: "BPRE >= 6 for last 2 weeks",
        21: "2 week Average BPRE >= 11",
        22: "BPRE >= 2 for last 3 weeks",
        23: "BPRE >= 4 for last 2 weeks",
        24: "2 week Average BPRE >= 9",
        25: "No rules satisfied"
    }

    for round_num in rounds:
        cumulative_data = consolidated_data[consolidated_data['Round'] <= round_num]
        player_data = cumulative_data[cumulative_data['Player'] == player_name]
        
        if player_data.empty:
            print(f"Round {round_num}: No data for player {player_name}")
            continue
        
        priority_level = assign_priority_level(player_data.iloc[-1], cumulative_data)
        rule_description = rule_descriptions.get(priority_level, "Unknown rule")
        print(f"Rule levels passed as at round {round_num}: Rule Level Satisfied: {priority_level} - {rule_description}")

if __name__ == "__main__":
    try:
        consolidated_data = load_data()  # Removed file_path
        print(f"Successfully loaded data for {consolidated_data['Round'].nunique()} rounds")
        
        # Get user preference for optimization strategy first
        while True:
            strategy = input("\nDo you want to:\n1. Maximize value (BPRE)\n2. Maximize base stats\n3. Hybrid approach (BPRE + Base stats)\nEnter 1, 2, or 3: ")
            if strategy in ['1', '2', '3']:
                break
            print("Invalid input. Please enter 1, 2, or 3.")

        maximize_base = (strategy == '2')
        hybrid_approach = (strategy == '3')

        # Then get position preferences
        valid_positions = ['HOK', 'HLF', 'CTR', 'WFB', 'EDG', 'MID']
        while True:
            print("\nSelect positions to consider:")
            print("0. All positions")
            for i, pos in enumerate(valid_positions, 1):
                print(f"{i}. {pos}")
            
            try:
                pos1 = int(input("\nSelect first position (0-6): "))
                if pos1 < 0 or pos1 > 6:
                    raise ValueError
                
                if pos1 == 0:
                    allowed_positions = None
                    break
                
                pos2 = int(input("Select second position (1-6, or same as first position): "))
                if pos2 < 1 or pos2 > 6:
                    raise ValueError
                
                allowed_positions = [valid_positions[pos1-1]]
                if pos1 != pos2:
                    allowed_positions.append(valid_positions[pos2-1])
                break
            except ValueError:
                print("Invalid input. Please enter valid numbers.")

        # Add P. Haas stats check for option 2
        if maximize_base:
            player_name = "P. Haas"
            avg_base = calculate_average_base(player_name, consolidated_data)
            latest_round = consolidated_data['Round'].max()
            latest_data = consolidated_data[
                (consolidated_data['Round'] == latest_round) & 
                (consolidated_data['Player'] == player_name)
            ]
            if not latest_data.empty:
                current_base = latest_data.iloc[0]['Total base']
                print(f"\n{player_name}'s stats:")
                print(f"Current base: {current_base:.0f}")
                print(f"Average base over last 3 rounds: {avg_base:.0f}\n")
        
        traded_out_players = ["Player1", "Player2"]  # Example players to trade out
        
        # Get lockout and simulation date/time preferences
        apply_lockout = input("Apply lockout restriction? (yes/no): ").strip().lower() == 'yes'
        simulate_datetime = None
        if apply_lockout:
            simulate_datetime = input("Enter simulated date/time (YYYY-MM-DDTHH:MM): ").strip()
        
        print(f"\nCalculating trade options for trading out: {', '.join(traded_out_players)}")
        print(f"Strategy: {'Maximizing base stats' if maximize_base else 'Maximizing value (BPRE)' if not hybrid_approach else 'Hybrid approach (BPRE + Base stats)'}")
        if allowed_positions:
            print(f"Considering only positions: {', '.join(allowed_positions)}")
        else:
            print("Considering all positions")
        
        options = calculate_trade_options(
            consolidated_data,
            traded_out_players,
            maximize_base=maximize_base,
            hybrid_approach=hybrid_approach,
            max_options=10,
            allowed_positions=allowed_positions,
            simulate_datetime=simulate_datetime,
            apply_lockout=apply_lockout
        )
        
        if options:
            print("\n=== Recommended Trade Combinations ===\n")
            for i, option in enumerate(options, 1):
                print(f"\nOption {i}")
                print("Players to trade in:")
                for player in option['players']:
                    if maximize_base:
                        print(f"- {player['name']} ({player['position']})")
                        print(f"  Price: ${player['price']:,}")
                        print(f"  Current Base: {player['total_base']}")
                        print(f"  Average Base: {player['projection']:.0f}")
                    else:
                        print(f"- {player['name']} ({player['position']})")
                        print(f"  Price: ${player['price']:,}")
                        print(f"  Current Base Premium: {player['diff']}")
                        print(f"  Consecutive Weeks above threshold: {player['consecutive_good_weeks']}")
                
                print(f"Total Price: ${option['total_price']:,}")
                if maximize_base:
                    print(f"Combined Projection: {option['total_projection']:.0f}")
                else:
                    print(f"Combined Diff: {option['total_diff']:.0f}")
                print(f"Salary Remaining: ${option['salary_remaining']:,}")
            
    except FileNotFoundError:
        print("Error: Could not find data file in the current directory")
    except ValueError as e:
        print("Error:", str(e))
    except Exception as e:
        print("An error occurred:", str(e))