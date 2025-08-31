import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_and_evaluate(models_dict, preproc, X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models with preprocessing pipeline.
    
    Parameters
    ----------
    models : dict
        Dictionary of model name -> sklearn estimator.
    preproc : transformer
        Preprocessing transformer (e.g., ColumnTransformer).
    X_train, y_train : training features and labels
    X_test, y_test : test features and labels
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame of metrics for each model.
    trained_pipelines : dict
        Dictionary of trained pipelines {model_name: fitted_pipeline}.
    """
    
    results = []
    trained_pipelines = {}

    for name, model in models_dict.items():
        # Build pipeline
        pipeline = Pipeline([
            ('preprocess', preproc),
            ('model', model)
        ])

        # Fit model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"]
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    # Results as DataFrame
    results_df = pd.DataFrame(results)
    return results_df, trained_pipelines
