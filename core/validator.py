import numpy as np
import pandas as pd


class NumericValidator:

    @staticmethod
    def validate(df):

        df_new = df.copy()
        issues = {}

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:

            min_val = df[col].min()

            if min_val < 0:
                issues[col] = f"Negative values detected (min={min_val})"

                # Business rule: clip negatives to 0
                df_new[col] = df[col].clip(lower=0)

        return df_new, issues