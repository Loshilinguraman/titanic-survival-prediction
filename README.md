
# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using Logistic Regression.

## Overview
This project uses the classic Kaggle Titanic dataset (891 passenger records) to build a binary classification model predicting whether a passenger survived, based on features like class, age, sex, fare, and family size aboard.

## Dataset
- Source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- 891 rows, 12 original columns
- Target variable: `Survived` (0 = did not survive, 1 = survived)

## Process
1. **Data Cleaning**
   - Filled missing `Age` and `Fare` values with median
   - Dropped `Cabin` (77% missing), `Name`, `Ticket`, and `PassengerId` (not predictive)
   - Filled missing `Embarked` values with the most frequent value
2. **Feature Encoding**
   - Converted `Sex` and `Embarked` from categorical text to numeric values
3. **Model Training**
   - Split data 80/20 into training and test sets
   - Trained a Logistic Regression model using scikit-learn
4. **Evaluation**
   - Achieved **79.9% accuracy** on the held-out test set

## Tools Used
Python, Pandas, Scikit-learn, Google Colab

## Results
The model correctly predicts survival outcomes for ~80% of passengers in the test set, outperforming the baseline "majority class" and simple gender-based heuristics.

## Future Improvements
- Try other models (Random Forest, XGBoost) for comparison
- Add feature engineering (e.g. extracting titles from names, family size grouping)
- Perform cross-validation for more robust accuracy estimates
