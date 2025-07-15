---

# 🎾 Tennis Ace  
🔍 Predicting ATP Player Winnings with Linear Regression

## 📌 Project Description  
This project analyzes professional tennis data to explore how different playing statistics relate to annual winnings. By developing single, two-feature, and multi-feature linear regression models, we investigate which performance metrics most strongly predict financial success for players ranked in the ATP.

The project uses Python, scikit-learn, and Matplotlib to explore, model, and visualize these relationships.

## 🎯 Project Objectives  
✅ Build regression models that predict outcomes based on ATP performance stats  
✅ Conduct exploratory data analysis (EDA) using Matplotlib visualizations  
✅ Evaluate model performance using R² score and visual comparisons  
✅ Compare different feature sets to determine the most predictive variables  
✅ Communicate findings and insights clearly through graphs and markdown commentary  

## 📂 Dataset Overview  
This project relies on ATP men’s tennis data between 2009 and 2017, focusing on the top 1500 ranked players each year. The dataset includes:

- **Offensive stats** → `Aces`, `DoubleFaults`, `FirstServePointsWon`, etc.  
- **Defensive stats** → `BreakPointsOpportunities`, `ReturnGamesWon`, etc.  
- **Outcomes** → `Wins`, `Losses`, `Winnings`, `Ranking`

File used:  
📄 `tennis_stats.csv` → Performance stats and outcome data per player/year

## 🔬 Analysis Breakdown

### 📊 Exploratory Analysis  
We analyzed the relationships between individual features and outcomes like `Win Percentage` and `Winnings` using scatter plots and histograms. Observations include:  
- **BreakPointsOpportunities** shows a strong linear relationship with `Winnings`  
- **DoubleFaults** may negatively impact earnings  
- **Aces** and `FirstServeReturnPointsWon` show weaker correlations individually

### 📈 Linear Regression Models
We developed three categories of models:

- **Single-feature models** → e.g., `BreakPointsOpportunities → Winnings`  
- **Two-feature models** → e.g., `BreakPointsOpportunities + FirstServeReturnPointsWon → Winnings`  
- **Multi-feature model** → Using 18 features to predict `Winnings`, achieving an **R² score of 0.8149**

### 💬 Summary of Insights  
- **Key Predictive Feature**: `BreakPointsOpportunities` consistently ranked as most influential  
- **Best Model**: Multi-feature model using both service and return stats  
- **Model Performance**: Visualizations confirm high alignment between predicted and actual `Winnings` in the best models

## ⚙️ Project Setup  

### 🖥️ Local Setup

1️⃣ Clone this repo:  
```bash
git clone git@personal-github:gabrielarcangelbol/tennis-ace.git
cd tennis-ace
```

2️⃣ Create a virtual environment (optional but recommended):  
```bash
python -m venv tennis-env
source tennis-env/bin/activate  # Mac/Linux  
.\tennis-env\Scripts\activate   # Windows
```

3️⃣ Install dependencies:  
```bash
pip install pandas matplotlib scikit-learn numpy
```

4️⃣ Run the notebook (if applicable):  
```bash
jupyter notebook
```

## 💻 Project Files  

📄 `tennis_ace_analysis.py` → Python script containing the full modeling pipeline  
📄 `tennis_stats.csv` → Raw dataset used for training and testing  
📄 `README.md` → Project overview and setup instructions  

## 🛠 Debugging Tips & Resources  
- ✅ Filter out invalid or zero data (`Wins + Losses == 0`) to prevent division errors  
- ✅ Use `.dropna()` and `.replace([inf], np.nan)` where needed  
- ✅ Modularize repetitive code blocks using helper functions (`train_and_evaluate()`)  
- 📚 Documentation:
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [Matplotlib Docs](https://matplotlib.org/stable/users/index.html)
  - [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## ✨ Future Work  
📌 Try regularization techniques (Ridge, Lasso) to compare performance  
📌 Explore nonlinear models like Decision Trees or Random Forests  
📌 Investigate ATP doubles data for team-level insights  

## 🔍 Contributing  
Have suggestions, model improvements, or ideas? Feel free to fork this repo, open issues, or submit pull requests! 💡

---

🎾 Let’s decode what makes a tennis champion — stat by stat.

---