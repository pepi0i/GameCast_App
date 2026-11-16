import sys
from pathlib import Path
import pandas as pd
import streamlit as st

#Making sure Python can find predictor.py (change "Model" if you put predictor.py in a different folder)
sys.path.append(str(Path(__file__).resolve().parent / "github")) #Adding the Model folder to the places Python looks for imports
from poisson_model import load_model, predict_match #Importing the two functions the page needs


#Training the model only once and reusing it on every click (without this, the model would retrain every time the page reruns)
@st.cache_resource(show_spinner="Loading and training the model...")
def get_model():
    return load_model() #Loading the data and training the Poisson models


model = get_model() #Getting the cached model
teams = model["current_teams"] #Teams in the league this season, spelled exactly like the data (e.g. "Man United")

st.title("GameCast Match Predictor")
st.write(
    "This project predicts match outcomes based on previous results"
)
home_team = st.selectbox("Home Team", teams, filter_mode=None)


away_team = st.selectbox("Away Team", teams, index=1, filter_mode=None) #Starting on a different team so the two boxes do not match by default


if home_team == away_team:
    st.warning("Please pick two different teams!")
else:
    st.success(f"Valid Matchup: {home_team} vs {away_team}")
if home_team != away_team and st.button("Predict Match", disabled=home_team == away_team):
    result = predict_match(model, home_team, away_team) #Calling the real prediction function from predictor.py

    for note in result["notes"]: #Looping through any warnings (e.g. a newly promoted team)
        st.warning(note) #Showing each warning

    c1, c2, c3 = st.columns(3) #Three columns for the three outcomes
    c1.metric(f"{home_team} win", f"{result['home_win']:.1%}") #Home win probability
    c2.metric("Draw", f"{result['draw']:.1%}") #Draw probability
    c3.metric(f"{away_team} win", f"{result['away_win']:.1%}") #Away win probability

    chart_data = pd.DataFrame({"Probability": [result["home_win"], result["draw"], result["away_win"]]}, index=[f"{home_team} win", "Draw", f"{away_team} win"]) #Building a small table for the bar chart
    st.bar_chart(chart_data) #Showing the three probabilities as a bar chart

    st.write(f"**Expected goals:** {home_team} {result['exp_home_goals']:.2f} - {result['exp_away_goals']:.2f} {away_team}") #Showing the expected goals for each side
    st.write(f"**Elo ratings:** {home_team} {result['home_elo']:.0f} | {away_team} {result['away_elo']:.0f}") #Showing both teams' Elo ratings
    st.write("**Most likely scores:** " + ", ".join(f"{s['score']} ({s['prob']:.1%})" for s in result["top_scores"])) #Showing the 3 most likely scorelines
