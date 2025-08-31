# Handle Outliers
## Detect Outliers
- All columns were checked for outliers except for `credit_policy`, `purpose`, and `fico`.
    + `credit_policy` and `purpose` are categorical columns, hence they are not included.
    + `fico` is not included as all values fall within the FICO range.
- IQR and z-score methods were used to detect outliers.
- The IQR multiplier and z-score threshold is determined based on the skew and kurtosis of the columns.
    + If the data was approximately normal, the IQR multiplier was set to 1.5, and the z-score threshold was set to 3.
    + If the data was moderately skewed, the IQR multiplier was set to 2.0, and the z-score threshold was set to 3.5.
    + If the data was heavily skewed, the IQR multiplier was set to 3.0, and the z-score threshold was set to `NaN` as z-score is not a reliable way to check for outliers in this case.

## Winsorization
+ It was decided not to drop any data, as the results showed a large number of outliers for multiple columns. This could be important information in understanding the loan defaults.
+ The complete dataset and a winsorized dataset was compared using their mean, standard deviation, skew, and kurtosis.
+ Winsorization was done to the top 1% of data only as there were no outliers in the lower part.
+ The final columns chosen to be winsorized had less percentage change in mean (~ <1%) and standard deviation (~ <10%).