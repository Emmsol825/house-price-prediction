# 🏠 House Price Prediction: Ames Housing

Predicting residential sale prices with **Gradient Boosting** on the Kaggle competition
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

> The dataset describes homes in **Ames, Iowa** (79 features per house). The target is `SalePrice`.

## 📌 Objective
Build a regression model that predicts the sale price of a house from its characteristics
(size, quality, location, age, garage, basement, etc.) and generate a Kaggle-ready submission file.

## 📊 Results

| Model (evaluated on an unseen 20 % hold-out of `train.csv`) | RMSLE ↓ | R² |
|---|---|---|
| **Gradient Boosting (tuned)** ✅ | **0.1150** | **0.922** |
| Gradient Boosting | 0.1220 | 0.912 |
| Ridge Regression | 0.1230 | 0.910 |
| Hist Gradient Boosting | 0.1363 | 0.890 |
| Random Forest | 0.1438 | 0.877 |
| Decision Tree | 0.2197 | 0.714 |

RMSLE (root mean squared log error) is the Kaggle metric. Lower is better.

## 🔧 Approach
1. **EDA**: target skewness, missing values, correlations, outliers, `LotFrontage` by neighbourhood.
2. **Cleaning**: two extreme outliers (very large, cheap houses) removed from training only; missing values handled with domain logic (e.g. no pool is not the same as unknown).
3. **Feature engineering**: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`; `LotFrontage` imputed with the neighbourhood median.
4. **Leak-free pipelines**: all preprocessing is fitted on training data only.
5. **Target transform**: `log1p(SalePrice)`, so RMSE on the target equals RMSLE.
6. **Model selection**: five models compared on an unseen 20 % hold-out; the best family tuned with `RandomizedSearchCV` (3-fold CV on the training portion only).
7. **Final model**: the winner is **retrained on all cleaned rows of `train.csv`** and used to predict `test.csv`.
8. **Submission**: predictions converted back to dollars with `expm1` and saved in the `sample_submission.csv` format (`Id`, `SalePrice`).

## 📁 Repository structure
```
house-price-prediction/
├── notebooks/
│   └── House_Price_Prediction_Best.ipynb   # full analysis with explanations and charts
├── outputs/
│   └── submission.csv                      # predictions for the 1,459 houses in test.csv
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## ▶️ How to run
1. Download `train.csv`, `test.csv` and `sample_submission.csv` from the
   [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
   and put them in a new folder called `data/` at the root of the repo (it is git-ignored).
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `notebooks/House_Price_Prediction_Best.ipynb` and run all cells.
   The predictions are written to `outputs/submission.csv`.

## 🛠️ Tech stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## 🔮 Possible improvements
Blend or stack several models, try XGBoost / LightGBM / CatBoost, add more engineered features, and use repeated K-fold cross-validation for a more stable score.
