import requests
import pandas as pd
import os

def fetch_fpl_data():
    print("Fetching data from FPL API...")
    
    # The main FPL API endpoint
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    # Make the request
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print("Data fetched successfully!")
        
        # The data we want is in the 'elements' key (these are the players)
        players_df = pd.DataFrame(data['elements'])
        
        # Select only the columns we need for now
        # (web_name is the player's short name, now_cost is price in tenths, etc.)
        columns_to_keep = [
            'id', 'web_name', 'element_type', 'now_cost', 
            'transfers_in_event', 'transfers_out_event', 
            'cost_change_event', 'selected_by_percent', 'total_points'
        ]
        
        # Keep only those columns
        players_df = players_df[columns_to_keep]
        
        # Create the data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        filepath = 'data/raw_players.csv'
        players_df.to_csv(filepath, index=False)
        
        print(f"Success! Data saved to {filepath}")
        print(f"Total players fetched: {len(players_df)}")
        
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

if __name__ == "__main__":
    fetch_fpl_data()

