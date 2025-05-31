import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train model on real team-level stats
@st.cache_resource
def train_model():
    data = {
        'Match point': [12, 12, 11, 8, 5, 3],
        'Net Game Win': [18, 14, 14, 4, -5, -13],
        'Kills': [466, 438, 406, 418, 392, 314],
        'Deaths': [348, 346, 320, 356, 381, 403],
        'Assists': [1134, 1122, 932, 982, 957, 772],
        'Gold': [1568701, 1733619, 1677433, 1689264, 1723914, 1482760],
        'Damage': [5586423, 6008254, 6052262, 6199686, 6227979, 6000999],
        'Lord Kills': [29, 48, 37, 47, 51, 25],
        'Tortoise Kills': [47, 45, 52, 60, 56, 41],
        'Tower Destroy': [130, 233, 212, 195, 190, 128],
        'Champion': [1, 0, 0, 0, 0, 0]  # only 1 champion
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=['Champion'])
    y = df['Champion']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

# Train model once and reuse
model, feature_names = train_model()

st.title("🏆 MPL S15 Champion Predictor")

st.markdown("Enter team statistics below (team-level only) for **up to 6 teams** and rank their predicted champion probabilities.")

num_teams = st.slider("How many teams to evaluate?", 1, 6, 3)
team_inputs = []

for i in range(num_teams):
    st.subheader(f"📊 Team {i+1} Stats Input")
    team_name = st.text_input(f"Team {i+1} Name", value=f"Team {i+1}", key=f"name{i}")
    col1, col2, col3 = st.columns(3)

    with col1:
        mp = st.number_input(f"{team_name} - Match Point", min_value=0, value=10, key=f"mp{i}")
        net = st.number_input(f"{team_name} - Net Game Win", value=5, key=f"net{i}")
        kills = st.number_input(f"{team_name} - Total Kills", value=400, key=f"kills{i}")
        deaths = st.number_input(f"{team_name} - Total Deaths", value=350, key=f"deaths{i}")

    with col2:
        assists = st.number_input(f"{team_name} - Total Assists", value=1000, key=f"assists{i}")
        gold = st.number_input(f"{team_name} - Gold Earned", value=1600000, key=f"gold{i}")
        damage = st.number_input(f"{team_name} - Total Damage", value=6000000, key=f"dmg{i}")

    with col3:
        lord = st.number_input(f"{team_name} - Lord Kills", value=30, key=f"lord{i}")
        turtle = st.number_input(f"{team_name} - Tortoise Kills", value=40, key=f"turtle{i}")
        towers = st.number_input(f"{team_name} - Tower Destroyed", value=150, key=f"tower{i}")

    input_vector = [mp, net, kills, deaths, assists, gold, damage, lord, turtle, towers]
    team_inputs.append((team_name, input_vector))

# Predict on all input teams
if st.button("🔍 Predict Champion Probability"):
    input_data = np.array([vec for _, vec in team_inputs])
    team_names = [name for name, _ in team_inputs]
    probs = model.predict_proba(input_data)[:, 1]

    df_result = pd.DataFrame({
        "Team": team_names,
        "Champion Probability (%)": (probs * 100).round(2)
    }).sort_values(by="Champion Probability (%)", ascending=False).reset_index(drop=True)

    st.success("🏅 Prediction Results")
    st.dataframe(df_result)
