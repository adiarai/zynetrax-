import pandas as pd


class DataQualityAnalyzer:

    @staticmethod
    def analyze(df):

        quality_report = {}

        for col in df.columns:

            total = len(df)
            missing = df[col].isnull().sum()
            missing_ratio = missing / total
            unique_ratio = df[col].nunique() / total

            report = {
                "missing_ratio": round(float(missing_ratio), 4),
                "unique_ratio": round(float(unique_ratio), 4)
            }

            # Numeric diagnostics
            if pd.api.types.is_numeric_dtype(df[col]):
                report["skewness"] = round(float(df[col].skew()), 4)
                report["variance"] = round(float(df[col].var()), 4)

            quality_report[col] = report

        return quality_report