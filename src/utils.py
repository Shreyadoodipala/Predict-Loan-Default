import pandas as pd

def change_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change column names of the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame whose columns need to be renamed.

    Returns:
    pd.DataFrame: A DataFrame with updated column names.
    """
    cols_mapping = {
        'credit.policy': 'credit_policy',
        'int.rate': 'interest_rate',
        'installment': 'installment',
        'log.annual.inc': 'log_annual_income',
        'dti': 'debt_income_ratio',
        'days.with.cr.line': 'days_with_credit_line',
        'revol.bal': 'revolve_balance',
        'revol.util': 'revolve_utilized',
        'inq.last.6mths': 'inquiries_last_6_mon',
        'delinq.2yrs': 'delinquent_2_yrs',
        'pub.rec': 'public_recs',
        'not.fully.paid': 'default'
    }
    return df.rename(columns=cols_mapping)

def check_numeric_columns(df: pd.DataFrame, exclude: list = ['purpose']):
    """Return True if all non-excluded columns are numeric, and list non-numeric cols."""
    cols_to_check = [col for col in df.columns if col not in exclude]
    non_numeric = [col for col in cols_to_check if not pd.api.types.is_numeric_dtype(df[col])]
    return len(non_numeric) == 0, non_numeric

def validate_loaded(original, reloaded):
    all_numeric, non_numeric_cols = check_numeric_columns(reloaded)
    checks = {
        'shape_equal': original.shape == reloaded.shape,
        'all_numeric_except_purpose': all_numeric,
        'non_numeric_columns': non_numeric_cols
    }
    return checks