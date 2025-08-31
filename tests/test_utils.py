import pandas as pd
import pytest
import src.utils as utils
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_change_col_names():
    # Input DataFrame with both mapped and unmapped columns
    df = pd.DataFrame({
        "credit.policy": [1, 0],
        "int.rate": [0.1, 0.2],
        "dti": [15.0, 20.0],
        "extra_col": ["a", "b"]   # unmapped column should remain unchanged
    })

    # Expected columns after renaming
    expected_columns = [
        "credit_policy",     # mapped
        "interest_rate",     # mapped
        "debt_income_ratio", # mapped
        "extra_col"          # unchanged
    ]

    # Apply the function
    result_df = utils.change_col_names(df)

    # Assert that the columns are renamed correctly
    assert list(result_df.columns) == expected_columns

    # Assert that the values remain the same
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        pd.DataFrame({
            "credit_policy": [1, 0],
            "interest_rate": [0.1, 0.2],
            "debt_income_ratio": [15.0, 20.0],
            "extra_col": ["a", "b"]
        })
    )

def test_empty_dataframe():
    # Edge case: empty DataFrame with only columns
    df = pd.DataFrame(columns=["credit.policy", "log.annual.inc"])
    result_df = utils.change_col_names(df)

    assert list(result_df.columns) == ["credit_policy", "log_annual_income"]
    assert result_df.empty
