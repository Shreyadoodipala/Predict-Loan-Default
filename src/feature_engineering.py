import numpy as np
import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the input DataFrame
    Args:
        df (pd.DataFrame): Raw input data.
    Returns:
        pd.DataFrame: DataFrame with new/engineered features.
    """
    df = df.copy()
    # Income to installment ratio
    if 'log_annual_income' in df.columns and 'installment' in df.columns:
        df['income_installment_ratio'] = df['log_annual_income'].apply(lambda x: np.exp(x)) / (df['installment'] * 12)
    # Debt to credit ratio
    if 'revolve_balance' in df.columns and 'log_annual_income' in df.columns:
        df['debt_to_credit_ratio'] = df['revolve_balance'] / df['log_annual_income'].apply(lambda x: np.exp(x))
    return df
