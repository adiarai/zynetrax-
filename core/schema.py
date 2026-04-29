import pandas as pd


class SmartSchemaDetector:

    @staticmethod
    def analyze_dataframe(df):

        schema = {}

        for col in df.columns:
            col_data = df[col]

            if pd.api.types.is_numeric_dtype(col_data):
                schema[col] = {
                    "type": "numeric",
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std())
                }

            elif pd.api.types.is_datetime64_any_dtype(col_data):
                schema[col] = {
                    "type": "datetime",
                    "min_date": str(col_data.min()),
                    "max_date": str(col_data.max())
                }

            elif col_data.nunique() < 20:
                schema[col] = {
                    "type": "categorical",
                    "unique_values": col_data.unique().tolist()
                }

            else:
                schema[col] = {
                    "type": "text",
                    "unique_count": int(col_data.nunique())
                }

        return schema