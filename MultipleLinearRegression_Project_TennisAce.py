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
# Load data from CSV into DataFrame
df = pd.read_csv('tennis_stats.csv')

# Display first few rows to check format
print(df.head())

# Show data types and column info
print(df.info())

# Check for missing values in each column
print(df.isnull().sum())

# ---------------------------------------------------
# 📊 STEP 2: Exploratory Data Analysis
# ---------------------------------------------------
# Print basic statistics summary for numeric features
print(df.describe())

# Calculate win percentage inline
win_pct = df['Wins'] / (df['Wins'] + df['Losses'])

# Filter for valid win percentages between 0 and 1
win_pct_clean = win_pct[(win_pct > 0) & (win_pct <= 1)]

# Histogram of win percentage
plt.figure(figsize=(8, 5))
plt.hist(win_pct_clean, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Win Percentage')
plt.xlabel('Win Percentage')
plt.ylabel('Number of Players')
plt.tight_layout()
plt.show()

# Filter valid data points: players with matches and non-null winnings
valid_data = df[(df['Wins'] + df['Losses'] > 0) & df['Winnings'].notnull()]
win_pct_valid = valid_data['Wins'] / (valid_data['Wins'] + valid_data['Losses'])

# Scatter plot: Aces vs Win Percentage
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['Aces'], win_pct_valid, alpha=0.5)
plt.title('Aces vs Win Percentage')
plt.xlabel('Aces')
plt.ylabel('Win Percentage')
plt.tight_layout()
plt.show()

# Add regression line to Aces vs Win Percentage
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

# Scatter plot: BreakPointsOpportunities vs Winnings
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['BreakPointsOpportunities'], valid_data['Winnings'], alpha=0.5, color='darkgreen')
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings ($)')
plt.tight_layout()
plt.show()

# Scatter plot: Aces vs Winnings
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['Aces'], valid_data['Winnings'], alpha=0.5, color='royalblue')
plt.title('Aces vs Winnings')
plt.xlabel('Aces')
plt.ylabel('Winnings ($)')
plt.tight_layout()
plt.show()

# Scatter plot: FirstServe% vs Winnings
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['FirstServe'], valid_data['Winnings'], alpha=0.5, color='purple')
plt.title('FirstServe Percentage vs Winnings')
plt.xlabel('FirstServe (%)')
plt.ylabel('Winnings ($)')
plt.tight_layout()
plt.show()

# Scatter plot: DoubleFaults vs Winnings
plt.figure(figsize=(8, 5))
plt.scatter(valid_data['DoubleFaults'], valid_data['Winnings'], alpha=0.5, color='tomato')
plt.title('DoubleFaults vs Winnings')
plt.xlabel('DoubleFaults')
plt.ylabel('Winnings ($)')
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 📈 STEP 3: Single Feature Linear Regression
# ---------------------------------------------------
# Select one feature and one outcome column
# Feature: FirstServeReturnPointsWon | Outcome: Winnings
feature = df[['FirstServeReturnPointsWon']]
outcome = df[['Winnings']]

# Split the data into training and testing sets (80% train / 20% test)
feature_train, feature_test, outcome_train, outcome_test = train_test_split(
    feature, outcome, train_size=0.8, random_state=1
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(feature_train, outcome_train)

# Evaluate model performance using R² score
score = model.score(feature_test, outcome_test)
print("Model R² Score (Test Set):", round(score, 4))

# Make predictions using the trained model
outcome_pred = model.predict(feature_test)

# Plot predicted vs actual outcome
plt.figure(figsize=(8, 5))
plt.scatter(outcome_test, outcome_pred, alpha=0.4, color='dodgerblue')
plt.plot([outcome_test.min(), outcome_test.max()],
         [outcome_test.min(), outcome_test.max()],
         color='red', linestyle='--', label='Perfect Prediction Line')
plt.title('Predicted vs Actual Winnings')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.legend()
plt.tight_layout()
plt.show()

"""
The blue dots in the scatter plot are widely dispersed and do not follow a clear pattern near the red dashed “Perfect Prediction Line.”

Most predicted values are clustered near the lower winnings range, which suggests that the model struggles with predicting high earnings.
"""

# Option 1: BreakPointsOpportunities → Winnings
feature1 = df[['BreakPointsOpportunities']]
outcome = df[['Winnings']]

feature1_train, feature1_test, outcome_train, outcome_test = train_test_split(feature1, outcome, train_size=0.8, random_state=1)
model1 = LinearRegression()
model1.fit(feature1_train, outcome_train)
score1 = model1.score(feature1_test, outcome_test)
print("BreakPointsOpportunities → Winnings R² Score:", round(score1, 4))

# Option 2: TotalPointsWon → Winnings
feature2 = df[['TotalPointsWon']]
feature2_train, feature2_test, outcome_train, outcome_test = train_test_split(feature2, outcome, train_size=0.8, random_state=1)
model2 = LinearRegression()
model2.fit(feature2_train, outcome_train)
score2 = model2.score(feature2_test, outcome_test)
print("TotalPointsWon → Winnings R² Score:", round(score2, 4))

# Option 3: Aces → Winnings
feature3 = df[['Aces']]
feature3_train, feature3_test, outcome_train, outcome_test = train_test_split(feature3, outcome, train_size=0.8, random_state=1)
model3 = LinearRegression()
model3.fit(feature3_train, outcome_train)
score3 = model3.score(feature3_test, outcome_test)
print("Aces → Winnings R² Score:", round(score3, 4))

## Plot: Predicted vs Actual Winnings using BreakPointsOpportunities
# Define feature and outcome for best model
best_feature = df[['BreakPointsOpportunities']]
outcome = df[['Winnings']]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(best_feature, outcome, train_size=0.8, random_state=1)

# Train the model
best_model = LinearRegression()
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Scatter plot of predicted vs actual winnings
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='mediumseagreen')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Prediction Line')
plt.title('BreakPointsOpportunities → Predicted vs Actual Winnings')
plt.xlabel('Actual Winnings ($)')
plt.ylabel('Predicted Winnings ($)')
plt.legend()
plt.tight_layout()
plt.show()
# ---------------------------------------------------
# 📊 STEP 4: Two Feature Linear Regression
# ---------------------------------------------------
# Define outcome
outcome = df[['Winnings']]

# Option 1: BreakPointsOpportunities + FirstServeReturnPointsWon
features_1 = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
f1_train, f1_test, y_train, y_test = train_test_split(features_1, outcome, train_size=0.8, random_state=1)
model_1 = LinearRegression()
model_1.fit(f1_train, y_train)
score_1 = model_1.score(f1_test, y_test)
print("Model 1 R² Score:", round(score_1, 4))


# Option 2: Aces + DoubleFaults
features_2 = df[['Aces', 'DoubleFaults']]
f2_train, f2_test, _, _ = train_test_split(features_2, outcome, train_size=0.8, random_state=1)
model_2 = LinearRegression()
model_2.fit(f2_train, y_train)
score_2 = model_2.score(f2_test, y_test)
print("Model 2 R² Score:", round(score_2, 4))


# Option 3: TotalPointsWon + ServiceGamesWon
features_3 = df[['TotalPointsWon', 'ServiceGamesWon']]
f3_train, f3_test, _, _ = train_test_split(features_3, outcome, train_size=0.8, random_state=1)
model_3 = LinearRegression()
model_3.fit(f3_train, y_train)
score_3 = model_3.score(f3_test, y_test)
print("Model 3 R² Score:", round(score_3, 4))

## Visualization: Model 1 – Predictions vs. Actual Winnings
# Use features from Model 1
# Generate predictions from model_1 on test set
y_pred_1 = model_1.predict(f1_test)

# Plot: actual vs predicted winnings
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_1, alpha=0.5, color='slateblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='crimson', linestyle='--', label='Perfect Prediction Line')
plt.title('Model 1: Predicted vs Actual Winnings')
plt.xlabel('Actual Winnings ($)')
plt.ylabel('Predicted Winnings ($)')
plt.legend()
plt.tight_layout()
plt.show()



# ---------------------------------------------------
# 🔢 STEP 5: Multiple Feature Linear Regression
# (To be implemented next)
# ---------------------------------------------------
## Multiple Feature Linear Regression: Model Creation
# Select multiple features from your dataset to predict Winnings
features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
               'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
               'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
               'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
               'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
               'TotalServicePointsWon']]

# Define outcome variable
outcome = df[['Winnings']]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(features, outcome, train_size=0.8, random_state=1)

# Create and train the model
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Evaluate model performance
multi_score = multi_model.score(X_test, y_test)
print("Multi-feature Model R² Score:", round(multi_score, 4))


## Optional: Plot Predictions vs Actual Winnings

# Predict Winnings on test set
y_pred = multi_model.predict(X_test)

# Plot prediction vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='navy')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='orange', linestyle='--', label='Perfect Prediction Line')
plt.title('Multiple Features → Predicted vs Actual Winnings')
plt.xlabel('Actual Winnings ($)')
plt.ylabel('Predicted Winnings ($)')
plt.legend()
plt.tight_layout()
plt.show()


"""
💬 Best Multi-Feature Model Report I created a linear regression model using 18 features related to both service and return game performance, including: FirstServe, FirstServePointsWon, FirstServeReturnPointsWon, SecondServePointsWon, SecondServeReturnPointsWon, Aces, BreakPointsConverted, BreakPointsFaced, BreakPointsOpportunities, BreakPointsSaved, DoubleFaults, ReturnGamesPlayed, ReturnGamesWon, ReturnPointsWon, ServiceGamesPlayed, ServiceGamesWon, TotalPointsWon, TotalServicePointsWon.

My model achieved an R² score of 0.8149 on the test data — the best performance across all models I tested.

💡 Key Insight: BreakPointsOpportunities stood out consistently as one of the strongest predictors of earnings, across both single and multi-feature models.

This suggests that players who generate more break opportunities tend to secure better results and earn higher winnings, emphasizing the strategic importance of strong return games at the professional level.
"""