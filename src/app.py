import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. SET PAGE TITLE
st.set_page_config(page_title="FPL Price Predictor", page_icon="⚽", layout="wide")

st.title("⚽ FPL Price Change Predictor")
st.write("This tool uses Machine Learning to predict if a player's price will rise or fall based on their transfer activity.")

# 2. LOAD AND PREPARE DATA (Cached so it doesn't reload every time)
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw_players.csv')
    
    # Recreate the features we made in the notebook
    df['net_transfers'] = df['transfers_in_event'] - df['transfers_out_event']
    df['selected_by_percent'] = df['selected_by_percent'].astype(float)
    df['transfer_velocity'] = df['net_transfers'] / (df['selected_by_percent'] + 0.1)
    
    # Recreate the target
    df['target'] = df['cost_change_event'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    return df

df = load_data()

# 3. TRAIN THE MODEL IN THE BACKGROUND
@st.cache_resource
def train_model(data):
    X = data[['net_transfers', 'transfer_velocity']]
    y = data['target']
    
    # Train on all data for the app
    model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# 4. BUILD THE USER INTERFACE
st.sidebar.header("Player Statistics")

# Let the user type a player's name
player_name = st.sidebar.selectbox("Select a Player", df['web_name'].sort_values())

# Get the data for the selected player
player_data = df[df['web_name'] == player_name].iloc[0]

# Display their current stats
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"£{player_data['now_cost'] / 10}m")
col2.metric("Net Transfers", f"{player_data['net_transfers']:,}")
col3.metric("Ownership", f"{player_data['selected_by_percent']}%")

# 5. MAKE A PREDICTION
if st.button("Predict Price Change"):
    # Prepare the input for the model
    input_features = pd.DataFrame({
        'net_transfers': [player_data['net_transfers']],
        'transfer_velocity': [player_data['transfer_velocity']]
    })
    
    prediction = model.predict(input_features)[0]
    
    if prediction == 1:
        st.success(f"📈 **{player_name}** is predicted to **RISE** in price!")
    elif prediction == -1:
        st.error(f"📉 **{player_name}** is predicted to **FALL** in price!")
    else:
        st.info(f"➡️ **{player_name}** is predicted to **STAY THE SAME**.")