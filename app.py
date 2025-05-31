import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load and train on existing MPL S15 top 6 data
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
        'Total Kills': [43, 40, 105, 23, 89, 2],
        'AVG Kills': [1.39, 1.25, 3.28, 0.74, 2.78, 1.0],
        'Total Deaths': [69, 102, 79, 148, 62, 5],
        'AVG Deaths': [2.23, 3.19, 2.47, 4.77, 1.94, 2.5],
        'Total Assists': [195, 147, 143, 178, 119, 6],
        'AVG Assists': [6.29, 4.59, 4.47, 5.74, 3.72, 3.0],
        'KDA Ratio': [3.45, 1.83, 3.14, 1.36, 3.35, 1.6],
        'Kill Participation': [0.7933, 0.6192, 0.8212, 0.686, 0.6887, 0.7273],
        'Champion': [1, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=['Champion'])
    y = df['Champion']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = train_model()

st.title("🏆 MPL S15 Champion Prediction")
st.markdown("Input stats for up to 6 teams to predict and rank their chances of becoming champion.")

num_teams = st.slider("How many teams to evaluate?", 1, 6, 3)
team_inputs = []

for i in range(num_teams):
  st.subheader(f"📊 Team {i+1} Stats Input")
    team_name = st.text_input(f"Team {i+1} Name", value=f"Team {i+1}", key=f"name{i}")
    cols = st.columns(3)

    with cols[0]:
        mp = st.number_input(f"{team_name} - Match Point", value=10, key=f"mp{i}")
        net = st.number_input(f"{team_name} - Net Game Win", value=5, key=f"net{i}")
        kills = st.number_input(f"{team_name} - Kills", value=400, key=f"kills{i}")
        deaths = st.number_input(f"{team_name} - Deaths", value=350, key=f"deaths{i}")
        assists = st.number_input(f"{team_name} - Assists", value=1000, key=f"assists{i}")
        gold = st.number_input(f"{team_name} - Gold", value=1600000, key=f"gold{i}")

    with cols[1]:
        dmg = st.number_input(f"{team_name} - Damage", value=6000000, key=f"dmg{i}")
        lord = st.number_input(f"{team_name} - Lord Kills", value=30, key=f"lord{i}")
        turtle = st.number_input(f"{team_name} - Tortoise Kills", value=40, key=f"turtle{i}")
        tower = st.number_input(f"{team_name} - Tower Destroy", value=150, key=f"tower{i}")
        t_kills = st.number_input(f"{team_name} - Player Total Kills", value=50, key=f"tkills{i}")
        a_kills = st.number_input(f"{team_name} - Player AVG Kills", value=2.0, key=f"akills{i}")

    with cols[2]:
        t_deaths = st.number_input(f"{team_name} - Player Total Deaths", value=80, key=f"tdeaths{i}")
        a_deaths = st.number_input(f"{team_name} - Player AVG Deaths", value=2.5, key=f"adeaths{i}")
        t_assists = st.number_input(f"{team_name} - Player Total Assists", value=150, key=f"tassists{i}")
        a_assists = st.number_input(f"{team_name} - Player AVG Assists", value=5.0, key=f"aassists{i}")
        kda = st.number_input(f"{team_name} - KDA Ratio", value=3.0, key=f"kda{i}")
        kp = st.number_input(f"{team_name} - Kill Participation", min_value=0.0, max_value=1.0, value=0.7, key=f"kp{i}")

    input_vector = [mp, net, kills, deaths, assists, gold, dmg, lord, turtle, tower,
                    t_kills, a_kills, t_deaths, a_deaths, t_assists, a_assists, kda, kp]

    team_inputs.append((team_name, input_vector))

# Prediction and Ranking
if st.button("🔍 Predict Champion Probability for All Teams"):
    input_data = np.array([vec for _, vec in team_inputs])
    team_names = [name for name, _ in team_inputs]
    probs = model.predict_proba(input_data)[:, 1]

    df_result = pd.DataFrame({
        "Team": team_names,
        "Champion Probability (%)": (probs * 100).round(2)
    }).sort_values(by="Champion Probability (%)", ascending=False).reset_index(drop=True)

    st.success("🏋️ Prediction Results")
    st.dataframe(df_result)
