import pandas as pd


class AdvancedColumnRoleDetector:

    @staticmethod
    def detect_roles(df):

        roles = {}

        for col in df.columns:

            unique_count = df[col].nunique()
            total_count = len(df)
            unique_ratio = unique_count / total_count
            null_ratio = df[col].isnull().sum() / total_count

            # True Primary Key
            if unique_count == total_count and null_ratio == 0:
                roles[col] = "Primary Key"

            # Strong Foreign Key pattern
            elif 0.1 < unique_ratio < 0.95:
                roles[col] = "Potential Foreign Key"

            # Numeric Feature
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].std() > 0:
                roles[col] = "Numeric Feature / Possible Target"

            # Low cardinality categorical
            elif unique_count < 20:
                roles[col] = "Categorical Feature"

            else:
                roles[col] = "Text / Identifier"

        return roles