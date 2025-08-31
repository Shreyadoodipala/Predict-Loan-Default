import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline

class FrequencyEncoder:
    def __init__(self):
        self.freq_maps = {}

    def fit(self, df, cols):
        """
        Learn frequency encoding maps from training data.
        
        Parameters:
            df (pd.DataFrame): training dataframe
            cols (list): list of categorical column names
        """
        for col in cols:
            self.freq_maps[col] = df[col].value_counts(normalize=True)
        return self

    def transform(self, df):
        """
        Apply learned frequency encoding to a dataframe.
        
        Parameters:
            df (pd.DataFrame): dataframe to transform
        """
        df = df.copy()
        for col, freq_map in self.freq_maps.items():
            df[col] = df[col].map(freq_map).fillna(0)
        return df

    def fit_transform(self, df, cols):
        """
        Fit and transform in one step (for training data).
        """
        self.fit(df, cols)
        return self.transform(df)
    
def build_preprocessor(X_train, categorical_cols, skew_thresholds=(1.0, 3.0)):
    """
    Build a preprocessing ColumnTransformer based on training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    categorical_cols : list
        List of categorical columns to ignore.
    skew_thresholds : tuple (mild, heavy)
        Thresholds for absolute skewness to define mild vs heavy skew.
    
    Returns
    -------
    preprocessor : ColumnTransformer
        Preprocessing pipeline (unfitted).
    col_groups : dict
        Dictionary with assigned column groups.
    """
    
    # Numerical columns = all numeric except categorical ones
    numerical_cols = [col for col in X_train.select_dtypes(include=[np.number]).columns 
                      if col not in categorical_cols]
    
    # Calculate skewness on training set
    skewness = X_train[numerical_cols].skew().fillna(0)
    
    normal_cols = skewness[skewness.abs() <= skew_thresholds[0]].index.tolist()
    mild_skewed_cols = skewness[(skewness.abs() > skew_thresholds[0]) & (skewness.abs() <= skew_thresholds[1])].index.tolist()
    heavy_skewed_cols = skewness[skewness.abs() > skew_thresholds[1]].index.tolist()
    
    # Define preprocessing
    preprocessor = ColumnTransformer([
        ('norm', StandardScaler(), normal_cols),
        ('mild', Pipeline([('pt_mild', PowerTransformer()), ('scaler', StandardScaler())]), mild_skewed_cols),
        ('heavy', Pipeline([('pt_heavy', PowerTransformer()), ('robust', RobustScaler())]), heavy_skewed_cols)
    ], remainder='passthrough')
    
    col_groups = {
        "categorical": categorical_cols,
        "normal": normal_cols,
        "mild_skewed": mild_skewed_cols,
        "heavy_skewed": heavy_skewed_cols
    }
    
    return preprocessor, col_groups

def preprocess_data(X_train, X_test, categorical_cols, skew_thresholds=(1.0, 3.0)):
    # Build preprocessor + col groups
    preprocessor, col_groups = build_preprocessor(X_train, categorical_cols, skew_thresholds)

    # Fit on train
    preprocessor.fit(X_train)

    # Transform train & test
    X_train_preproc = preprocessor.transform(X_train)
    X_test_preproc = preprocessor.transform(X_test)

    # Clean feature names
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.split("__")[-1] for name in feature_names]

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_preproc, columns=clean_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_preproc, columns=clean_names, index=X_test.index)

    return X_train_df, X_test_df, preprocessor, col_groups
