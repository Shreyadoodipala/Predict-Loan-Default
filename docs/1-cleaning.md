# Data Cleaning Summary

1. **Environment Setup**
   - Loaded environment variables using `dotenv`.
   - Imported required libraries: pandas, os, pathlib.

2. **Data Loading**
   - Loaded the raw loan data from the path specified by the environment variable `DATA_DIR_RAW`.
   - Used pandas to read the CSV file `loan_data.csv`.

3. **Column Name Cleaning**
   - Renamed columns for clarity and consistency:
     - `credit.policy` → `credit_policy`
     - `int.rate` → `interest_rate`
     - `log.annual.inc` → `log_annual_income`
     - `dti` → `debt_income_ratio`
     - `days.with.cr.line` → `days_with_credit_line`
     - `revol.bal` → `revolve_balance`
     - `revol.util` → `revolve_utilized`
     - `inq.last.6mths` → `inquiries_last_6_mon`
     - `delinq.2yrs` → `delinquent_2_yrs`
     - `pub.rec` → `public_recs`
     - `not.fully.paid` → `default`

4. **Data Preview**
   - Displayed the first few rows of the cleaned DataFrame to verify changes.

5. **Saving Cleaned Data**
   - Saved the cleaned DataFrame to a new CSV file: `loan_data_cols_changed.csv` in the raw data directory.

6. **Validation**
   - Reloaded the saved CSV to ensure data integrity.
   - Used a utility function (`utils.validate_loaded`) to compare the original and saved DataFrames.
   - Printed validation results to confirm successful cleaning and saving.
