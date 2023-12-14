"""
    Braden Harris
    CS-470
    Term Project
    Program loads in information from the 2023 season, as well as 
        a model saved in file. It asks the user for the week they 
        would like to predict.
"""
import numpy as np
import pandas as pd
import pickle
import neurolab as nl
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Load offense, defense, and game results data 23
o23 = pd.read_csv('Offense23.csv', skiprows=0)
d23 = pd.read_csv('Defense23.csv', skiprows=0)
results23 = pd.read_csv('Results23.csv')
combined_data23 = pd.merge(o23, d23, on='School', how='inner')

# Define the features to be used for prediction
selected_features = ['OPts', 'OPPct', 'OPYds', 'ORAvg', 'OPlays', 'OTot', 'DPts', 'DPPct', 'DPYds', 'DRAvg', 'DPlays', 'DYds', 'DTO']


def predict_game(home_team, away_team, combined_data, model, selected_features):
    # Find the stats for the two teams
    team1_stats = combined_data[combined_data['School'] == home_team][selected_features].values
    team2_stats = combined_data[combined_data['School'] == away_team][selected_features].values

    # Normalize the input data
    min_max_scaler = preprocessing.MinMaxScaler()
    team1_stats = min_max_scaler.fit_transform(team1_stats)
    team2_stats = min_max_scaler.transform(team2_stats)

    # Combine the two teams' stats
    input_data = np.concatenate((team1_stats, team2_stats), axis=None)

    # Reshape input_data
    input_data = input_data.reshape(1, -1)

    # Make prediction using the model
    prediction = (model.sim(input_data) >= 0.5).astype(int)

    if prediction == 0:
        winner = away_team
    else:
        winner = home_team

    return winner

# Iterates through the 2023 season predicting the desired games
def predict_games23(results23, combined_data23, model, selected_features):
    predictions = []

    for i in range(len(results23)):
        home_team = results23.iloc[i]['Home Team']
        away_team = results23.iloc[i]['Away Team']
        predicted_winner = predict_game(home_team, away_team, combined_data23, model, selected_features)
        predictions.append(predicted_winner)

    return predictions

def load_model_from_pickle(filename):
    try:
        with open(filename, 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model

    except FileNotFoundError:
        print("File not found:", filename)
    except Exception as e:
        print("An error occurred while loading the model:", str(e))
    return None

# Train the model
model = load_model_from_pickle('trained_model3.pkl')

# Ask the user for the week they want to predict
userInput = int(input("Enter which week of games you would like to predict(1-13): "))

# Filter the results dataframe to include only the games for the specified weeks
filtered_results23 = results23[(results23['Week'] == userInput)]

filtered_results23 = filtered_results23.reset_index(drop=True)

# Predict the outcomes for the filtered games
filtered_predictions_2023 = predict_games23(filtered_results23, combined_data23, model, selected_features)

# Evaluate the accuracy of the model on the training data
input_data = []

for i in range(len(filtered_results23)):
    home_team = filtered_results23.loc[i, 'Home Team']
    away_team = filtered_results23.loc[i, 'Away Team']

    team1_stats = combined_data23[combined_data23['School'] == home_team][selected_features].values
    team2_stats = combined_data23[combined_data23['School'] == away_team][selected_features].values

    input_data.append(np.concatenate((team1_stats, team2_stats), axis=None))

input_data = np.array(input_data)

# Display the results for the desired games
for i in range(len(filtered_results23)):
    home_team = filtered_results23.iloc[i]['Home Team']
    away_team = filtered_results23.iloc[i]['Away Team']
    predicted_winner = filtered_predictions_2023[i]
    week = filtered_results23.iloc[i]['Week']
    print(f"2023 Predicted winner of WeeK {week} - {home_team} vs. {away_team}: {predicted_winner}")