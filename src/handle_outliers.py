import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

def assign_thresholds(skew, kurt):
    """
    Assigns IQR (k) and z-score threshold based on skewness and kurtosis.
    """
    # Near-normal
    if abs(skew) < 1 and abs(kurt) < 3:
        return 1.5, 3  # standard boxplot + z-score
    
    # Moderate skew / heavy tails
    elif abs(skew) < 3 and abs(kurt) < 20:
        return 2.0, 3.5  # relaxed IQR, modified z-score
    
    # Extreme skew/heavy tails
    else:
        return 3.0, np.nan  # z-scores unreliable
    
def build_thresholds_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Build a DataFrame of skewness, kurtosis, and suggested IQR/Z-score thresholds
    for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list
        List of column names to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with skewness, kurtosis, IQR k, and z-score thresholds.
    """
    thresholds = {}
    for col in cols:
        skew = df[col].skew()
        kurt = df[col].kurt()
        k_val, z_val = assign_thresholds(skew, kurt)
        thresholds[col] = {
            "skewness": skew,
            "kurtosis": kurt,
            "iqr_k": k_val,
            "zscore_threshold": z_val
        }

    return pd.DataFrame(thresholds).T
    
def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean mask where True indicates an outlier by IQR rule.
    Parameters
    ----------
    series : pd.Series
        Numeric series to evaluate.
    k : float
        Multiplier for IQR to set fences (default 1.5).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask where True indicates |z| > threshold."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    z = (series - mu) / (sigma if sigma != 0 else 1.0)
    return z.abs() > threshold

def analyze_outliers(df: pd.DataFrame, thresholds_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Detect outliers for multiple columns using IQR and z-score methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    thresholds_df : pd.DataFrame
        DataFrame with thresholds (must include 'iqr_k' and 'zscore_threshold').
        Index should align with columns in `df`.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame showing counts of outliers per method and column.
    """
    outlier_results = {}

    for col in thresholds_df.index:
        series = df[col]

        # Get thresholds for this column
        k_val = thresholds_df.loc[col, "iqr_k"]
        z_val = thresholds_df.loc[col, "zscore_threshold"]

        # Apply IQR rule
        outliers_iqr = detect_outliers_iqr(series, k=k_val)

        # Apply z-score rule (only if threshold is not NaN)
        if not pd.isna(z_val):
            outliers_z = detect_outliers_zscore(series, threshold=z_val)
        else:
            outliers_z = pd.Series([False] * len(series), index=series.index)

        # Store masks
        outlier_results[col] = {
            "iqr_outliers": outliers_iqr,
            "zscore_outliers": outliers_z,
            "combined_outliers": outliers_iqr | outliers_z
        }

    # Build summary DataFrame
    summary = pd.DataFrame({
        col: {
            "IQR count": outlier_results[col]["iqr_outliers"].sum(),
            "Z-score count": outlier_results[col]["zscore_outliers"].sum(),
            "Combined count": outlier_results[col]["combined_outliers"].sum()
        }
        for col in outlier_results
    }).T

    return summary


def compare_winsorization(df: pd.DataFrame, cols: list, win_limits=(0, 0.01)):
    """
    Compare effect of winsorization on specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols : list
        List of columns to winsorize.
    target : str
        Compare model performance (Logistic Regression).
    win_limits : tuple
        Winsorization limits (default = (0, 0.01) → cap top 1%).
    """

    stats = {}
    df_winsor = df.copy()

    # Winsorize selected columns
    for col in cols:
        df_winsor[col] = winsorize(df[col], limits=win_limits)

        # Collect summary stats
        stats[col] = {
            "mean_raw": df[col].mean(),
            "mean_winsor": df_winsor[col].mean(),
            "std_raw": df[col].std(),
            "std_winsor": df_winsor[col].std(),
            "skew_raw": df[col].skew(),
            "skew_winsor": df_winsor[col].skew(),
            "kurt_raw": df[col].kurt(),
            "kurt_winsor": df_winsor[col].kurt()
        }

    stats_df = pd.DataFrame(stats).T
    stats_df['mean_change_pct'] = (stats_df['mean_winsor'] - stats_df['mean_raw']) / stats_df['mean_raw'] * 100
    stats_df['std_change_pct'] = (stats_df['std_winsor'] - stats_df['std_raw']) / stats_df['std_raw'] * 100

    return stats_df, df_winsor

def winsorize_df(df: pd.DataFrame, cols: list, lower: float = 0.00, upper: float = 0.99) -> pd.DataFrame:
    """
    Winsorize specified columns of a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list
        List of column names to winsorize.
    lower : float
        Lower quantile cutoff (default 0.00).
    upper : float
        Upper quantile cutoff (default 0.99 = 99%).

    Returns
    -------
    pd.DataFrame
        DataFrame with winsorized columns (copy).
    """
    df_wins = df.copy()
    for col in cols:
        if col in df_wins.columns:
            low_val = df_wins[col].quantile(lower)
            high_val = df_wins[col].quantile(upper)
            df_wins[col] = df_wins[col].clip(lower=low_val, upper=high_val)
    return df_wins