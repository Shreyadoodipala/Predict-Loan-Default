# Preprocessing
## Train-test split
- The data was split into 80% training data and 20% testing data, with stratification to ensure that both splits have an equal proportion of loans defaulted and not defaulted.

## Encoding categorical variables
- Frequency encoding was used to convert the `purpose` into numeric values.
- No further encoding was required for `credit_policy` and `default` as they're already numeric.

## Scaling numeric variables
The 'build_processor` function creates a preprocessing pipeline for scaling numeric variables based on their skewness:
- Identifies numeric columns: Selects all numeric features except the ones explicitly listed as categorical.
- Measures skewness: Calculates the skewness of each numeric column in the training set.
- Classifies features into groups:
    + Normal: skewness ≤ mild threshold → scaled with StandardScaler.
    + Mildly skewed: skewness between mild and heavy thresholds → transformed with PowerTransformer (to reduce skew) then StandardScaler.
    + Heavily skewed: skewness above heavy threshold → transformed with PowerTransformer then scaled with RobustScaler (less sensitive to outliers).
- Builds a ColumnTransformer: Automatically applies the appropriate transformation to each group while leaving other columns unchanged (remainder='passthrough').
- It returns preprocessor: the unfitted pipeline ready to transform training/testing data, and col_groups, a dictionary showing which columns were categorized as normal, mildly skewed, heavily skewed, or categorical.