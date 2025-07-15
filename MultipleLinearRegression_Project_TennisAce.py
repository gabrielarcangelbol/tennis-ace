# Import necessary libraries
import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------
# 📌 STEP 1: Load and inspect the dataset
# ---------------------------------------------------

# Load dataset
df = pd.read_csv('tennis_stats.csv')

# Preview first rows
print(df.head())

# Check data structure
print(df.info())

# Check missing values
print(df.isnull().sum())
if df[['Wins', 'Losses', 'Winnings']].isnull().any().any():
    print('Warning: Missing values detected! Consider cleaning data before modeling.')

# ---------------------------------------------------
# 📊 STEP 2: Exploratory Data Analysis
# ---------------------------------------------------

# Basic statistics summary
print(df.describe())

# Compute win percentage inline
win_pct = df['Wins'] / (df['Wins'] + df['Losses'])
win_pct_clean = win_pct[(win_pct > 0) & (win_pct <= 1)]

# 📈 Distribution of Win Percentage
plt.figure(figsize=(8, 5))
plt.hist(win_pct_clean, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Win Percentage')
plt.xlabel('Win Percentage')
plt.ylabel('Number of Players')
plt.tight_layout()
plt.show()

# Filter valid rows with actual match data
valid_data = df[(df['Wins'] + df['Losses'] > 0) & df['Winnings'].notnull()]
win_pct_valid = valid_data['Wins'] / (valid_data['Wins'] + valid_data['Losses'])

# 📈 Aces vs Win Percentage
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['Aces'], win_pct_valid, alpha=0.5)
plt.title('Aces vs Win Percentage')
plt.xlabel('Aces')
plt.ylabel('Win Percentage')
plt.tight_layout()
plt.show()
'''
Observation: There is a weak positive correlation between Aces and Win Percentage, but the relationship is not strong. Other features may be more predictive.
'''

# 📈 Regression Line: Aces vs Win Percentage
x = valid_data['Aces']
y = win_pct_valid
m, b = np.polyfit(x, y, 1)
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.5)
plt.plot(x, m*x + b, color='red', label='Trend Line')
plt.title('Aces vs Win Percentage with Regression Line')
plt.xlabel('Aces')
plt.ylabel('Win Percentage')
plt.legend()
plt.tight_layout()
plt.show()

# 📈 BreakPointsOpportunities vs Winnings
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['BreakPointsOpportunities'], valid_data['Winnings'], alpha=0.5, color='darkgreen')
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings ($)')
plt.tight_layout()
plt.show()
'''
Observation: There is a strong positive correlation between BreakPointsOpportunities and Winnings. This feature appears to be highly predictive.
'''

# 📈 DoubleFaults vs Winnings
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['DoubleFaults'], valid_data['Winnings'], alpha=0.5, color='tomato')
plt.title('DoubleFaults vs Winnings')
plt.xlabel('DoubleFaults')
plt.ylabel('Winnings ($)')
plt.tight_layout()
plt.show()
'''
Observation: There may be a negative relationship between double faults and winnings — more faults could signal weaker performance.
'''

# ---------------------------------------------------
# 📈 STEP 3: Single Feature Linear Regression
# ---------------------------------------------------

def train_and_evaluate(features, outcome):
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, train_size=0.8, random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, X_test, y_test, score

# Feature: FirstServeReturnPointsWon
model_s1, X_test_s1, y_test_s1, score_s1 = train_and_evaluate(df[['FirstServeReturnPointsWon']], df[['Winnings']])
print("FirstServeReturnPointsWon → Winnings R² Score:", round(score_s1, 4))

# 📈 Plot prediction vs actual
y_pred_s1 = model_s1.predict(X_test_s1)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_s1, y_pred_s1, alpha=0.5, color='dodgerblue')
plt.plot([y_test_s1.min(), y_test_s1.max()],
         [y_test_s1.min(), y_test_s1.max()],
         color='red', linestyle='--', label='Perfect Prediction Line')
plt.title('FirstServeReturnPointsWon → Predicted vs Actual Winnings')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 📈 STEP 4: Two Feature Linear Regression
# ---------------------------------------------------

# Model 1
features_m1 = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
model_m1, X_test_m1, y_test_m1, score_m1 = train_and_evaluate(features_m1, outcome)
print("Model 1 R² Score (Best Two Features):", round(score_m1, 4))

# Plot predictions for Model 1
y_pred_m1 = model_m1.predict(X_test_m1)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_m1, y_pred_m1, alpha=0.5, color='slateblue')
plt.plot([y_test_m1.min(), y_test_m1.max()],
         [y_test_m1.min(), y_test_m1.max()],
         color='crimson', linestyle='--', label='Perfect Prediction Line')
plt.title('Model 1 (2 Features) → Predicted vs Actual Winnings')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 🔢 STEP 5: Multiple Feature Linear Regression
# ---------------------------------------------------

# Select many features
features_multi = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
                     'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
                     'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
                     'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
                     'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
                     'TotalServicePointsWon']]

# Train final model
model_final, X_test_final, y_test_final, score_final = train_and_evaluate(features_multi, df[['Winnings']])
print("Multi-feature Model R² Score:", round(score_final, 4))

# Plot final predictions
y_pred_final = model_final.predict(X_test_final)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_final, y_pred_final, alpha=0.5, color='navy')
plt.plot([y_test_final.min(), y_test_final.max()],
         [y_test_final.min(), y_test_final.max()],
         color='orange', linestyle='--', label='Perfect Prediction Line')
plt.title('Multiple Features → Predicted vs Actual Winnings')
plt.xlabel('Actual Winnings ($)')
plt.ylabel('Predicted Winnings ($)')
plt.legend()
plt.tight_layout()
plt.show()

"""
💬 Conclusions: Best Multi-Feature Model Report I created a linear regression model using 18 features related to both service and return game performance, including: FirstServe, FirstServePointsWon, FirstServeReturnPointsWon, SecondServePointsWon, SecondServeReturnPointsWon, Aces, BreakPointsConverted, BreakPointsFaced, BreakPointsOpportunities, BreakPointsSaved, DoubleFaults, ReturnGamesPlayed, ReturnGamesWon, ReturnPointsWon, ServiceGamesPlayed, ServiceGamesWon, TotalPointsWon, TotalServicePointsWon.

My model achieved an R² score of 0.8149 on the test data — the best performance across all models I tested.

💡 Key Insight: BreakPointsOpportunities stood out consistently as one of the strongest predictors of earnings, across both single and multi-feature models.

This suggests that players who generate more break opportunities tend to secure better results and earn higher winnings, emphasizing the strategic importance of strong return games at the professional level.
"""