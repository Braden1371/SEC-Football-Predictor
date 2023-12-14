"""
    Braden Harris
    CS-470
    Term Project
    Program creates and trains a model to predict the winner of 
        SEC football games.The model can be saved to use later.
"""
import numpy as np
import pandas as pd
import pickle
import neurolab as nl
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Load offense, defense, and game results data 22
o22 = pd.read_csv('Offense22.csv', skiprows=0)
d22 = pd.read_csv('Defense22.csv', skiprows=0)
results22 = pd.read_csv('Results22.csv')
combined_data22 = pd.merge(o22, d22, on='School', how='inner')

# Load offense, defense, and game results data 21
o21 = pd.read_csv('Offense21.csv', skiprows=0)
d21 = pd.read_csv('Defense21.csv', skiprows=0)
results21 = pd.read_csv('Results21.csv')
combined_data21 = pd.merge(o22, d22, on='School', how='inner')

# Define features
selected_features = ['OPts', 'OPPct', 'OPYds', 'ORAvg', 'OPlays', 'OTot', 'DPts', 'DPPct', 'DPYds', 'DRAvg', 'DPlays', 'DYds', 'DTO']

def train_model(game_data21, game_data22, combined_data21, combined_data22, selected_features):
    input_size = len(selected_features) * 2  # Number of input neurons (doubled for the two teams)
    output_size = 1  # Number of output neurons

    input_range = [[0, 500] for _ in range(input_size)]

    model = nl.net.newff(input_range, [13, 6, output_size])  # Two hidden layers

    #training algorithm can be determined here
    train_algo = nl.train.train_gdx
    #train_algo = nl.train.train_gd

    input_data = []
    target_data = []

    for i in range(len(game_data22)):
        home_team = game_data22.loc[i, 'Home Team']
        away_team = game_data22.loc[i, 'Away Team']

        # Find the statistics for the two teams for the current game
        team1_stats = combined_data22[combined_data22['School'] == home_team][selected_features].values
        team2_stats = combined_data22[combined_data22['School'] == away_team][selected_features].values

        # Combine the two teams' statistics
        input_data.append(np.concatenate((team1_stats, team2_stats), axis=None))

        # Determine the winner based on the 'Winner' column in game_data
        target_data.append(game_data22.loc[i, 'Winner'])

    for i in range(len(game_data21)):
        home_team = (game_data21.loc[i, 'Home Team'])
        away_team = (game_data21.loc[i, 'Away Team'])

        # Find the statistics for the two teams for the current game
        team1_stats = (combined_data21[combined_data21['School'] == home_team][selected_features].values)
        team2_stats = (combined_data21[combined_data21['School'] == away_team][selected_features].values)

        # Combine the two teams' statistics
        input_data.append(np.concatenate((team1_stats, team2_stats), axis=None))

        # Determine the winner based on the 'Winner' column in game_data
        target_data.append(game_data21.loc[i, 'Winner'])

    # Normalize the input data
    min_max_scaler = preprocessing.MinMaxScaler()
    input_data = min_max_scaler.fit_transform(input_data)

    input_data = np.array(input_data)
    target_data = np.array(target_data).reshape(-1, 1)

    epochs = 200

    error_progress = train_algo(model, input_data, target_data, epochs=epochs, goal=0.000001)

    return model, error_progress

def predict_game(home_team, away_team, combined_data, model, selected_features):
    # Find the statistics for the two teams
    team1_stats = combined_data[combined_data['School'] == home_team][selected_features].values
    team2_stats = combined_data[combined_data['School'] == away_team][selected_features].values

    # Normalize the input data
    min_max_scaler = preprocessing.MinMaxScaler()
    team1_stats = min_max_scaler.fit_transform(team1_stats)
    team2_stats = min_max_scaler.transform(team2_stats)

    # Combine the two teams' statistics
    input_data = np.concatenate((team1_stats, team2_stats), axis=None)

    # Reshape input_data to match the model's input size
    input_data = input_data.reshape(1, -1)

    # Make a prediction using the trained network
    prediction = (model.sim(input_data) >= 0.5).astype(int)

    if prediction == 0:
        winner = away_team
    else:
        winner = home_team

    return winner

def predict_all_games(results21, results22, combined_data21, combined_data22, model, selected_features):
    predictions = []
    
    #go through entire 2022 and 2021 seasons and predict the winners using predict game function
    for i in range(len(results22)):
        home_team = results22.iloc[i]['Home Team']
        away_team = results22.iloc[i]['Away Team']
        predicted_winner = predict_game(home_team, away_team, combined_data22, model, selected_features)
        predictions.append(predicted_winner)

    for i in range(len(results21)):
        home_team = results21.iloc[i]['Home Team']
        away_team = results21.iloc[i]['Away Team']
        predicted_winner = predict_game(home_team, away_team, combined_data21, model, selected_features)
        predictions.append(predicted_winner)
    
    return predictions

#save desirable models as a pickle
def save_model_if_accuracy_above_threshold(model, accuracy, threshold):
    if accuracy > threshold:
        with open('trained_model4.pkl', 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved with accuracy {accuracy * 100:.2f}%")
    else:
        print(f"Model accuracy ({accuracy * 100:.2f}%) does not meet the threshold. Model not saved.")


# Train the model
model, error_progress = train_model(results21, results22, combined_data21, combined_data22, selected_features)

# Evaluate the accuracy of the model on the training data
input_data = []
target_data = []

#get game results and stats for the 2022 season
for i in range(len(results22)):
    home_team = results22.loc[i, 'Home Team']
    away_team = results22.loc[i, 'Away Team']

    team1_stats = combined_data22[combined_data22['School'] == home_team][selected_features].values
    team2_stats = combined_data22[combined_data22['School'] == away_team][selected_features].values

    input_data.append(np.concatenate((team1_stats, team2_stats), axis=None))
    target_data.append(results22.loc[i, 'Winner'])

#get game results and stats for 2021 season
for i in range(len(results21)):
    home_team = results21.loc[i, 'Home Team']
    away_team = results21.loc[i, 'Away Team']

    team1_stats = combined_data21[combined_data21['School'] == home_team][selected_features].values
    team2_stats = combined_data21[combined_data21['School'] == away_team][selected_features].values

    input_data.append(np.concatenate((team1_stats, team2_stats), axis=None))
    target_data.append(results21.loc[i, 'Winner'])

#store input and target data as arrays
input_data = np.array(input_data)
target_data = np.array(target_data).reshape(-1, 1)

# test the model on the input data
predictions = (model.sim(input_data) >= 0.5).astype(int)

#determine the accuracy of predictions vs actual
accuracy = accuracy_score(target_data, predictions)

# Predict the outcomes for all games
all_game_predictions = predict_all_games(results21,results22, combined_data21, combined_data22, model, selected_features)

# Display the results
for i in range(len(results22)):
    home_team = results22.iloc[i]['Home Team']
    away_team = results22.iloc[i]['Away Team']
    predicted_winner = all_game_predictions[i]
    print(f"2022 Predicted winner of {home_team} vs. {away_team}: {predicted_winner}")

for i in range(len(results21)):
    home_team = results21.iloc[i]['Home Team']
    away_team = results21.iloc[i]['Away Team']
    predicted_winner = all_game_predictions[i+(len(results22))]
    print(f"2021 Predicted winner of {home_team} vs. {away_team}: {predicted_winner}")

print(f"Model accuracy on training data: {accuracy * 100:.2f}%")

#saves model if accuracy is high enough
save_model_if_accuracy_above_threshold(model, accuracy, 0.80)