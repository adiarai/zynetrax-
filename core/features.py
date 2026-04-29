import numpy as np
import pandas as pd


class FeatureEngineer:

    @staticmethod
    def engineer(df):

        df_new = df.copy()

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:

            skewness = df[col].skew()

            # If highly skewed → apply log transform safely
            if abs(skewness) > 2:

                # Avoid negative issues before log
                min_val = df[col].min()

                if min_val < 0:
                    shifted = df[col] - min_val
                    df_new[f"{col}_log"] = np.log1p(shifted)
                else:
                    df_new[f"{col}_log"] = np.log1p(df[col])

        return df_new