# Modelling and Evaluation

## Setup
- Loaded environment variables and set up paths for processed data and models.
- Imported required libraries
- Loaded preprocessed training and test datasets.
- Loaded the preprocessor pipeline.

## Model Training
- Defined three classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Built pipelines combining preprocessing and each model.
- Trained each pipeline on the training data.
- Evaluated each model on the test data using metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Plotted confusion matrices for each model.
- Compiled results into a DataFrame and saved to `results/model_comparison.csv`.

## Evaluation and Model Selection
- Decision Tree was selected as the best model based on the highest F1 score.
- Saved the trained Decision Tree pipeline.
