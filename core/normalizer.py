import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class AdvancedDataNormalizer:

    @staticmethod
    def normalize(df):

        # -----------------------------
        # Basic Cleanup
        # -----------------------------
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        df.columns = df.columns.str.strip()

        # -----------------------------
        # Smart Datetime Detection (Pandas 2.x Safe)
        # -----------------------------
        for col in df.columns:

            if df[col].dtype == "object":

                sample_values = df[col].dropna().astype(str).head(10)

                # Only attempt conversion if values look like dates
                if not sample_values.str.contains(r"\d{4}-\d{2}-\d{2}", regex=True).any():
                    continue

                try:
                    converted = pd.to_datetime(df[col], errors="coerce")

                    # Convert only if majority parsed successfully
                    if converted.notna().sum() > len(df) * 0.7:
                        df[col] = converted

                except Exception:
                    continue

        # -----------------------------
        # Missing Value Handling
        # -----------------------------
        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("N/A")

        # -----------------------------
        # ML-Based Outlier Detection
        # -----------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 1:

            try:
                iso = IsolationForest(
                    contamination=0.02,
                    random_state=42
                )

                preds = iso.fit_predict(df[numeric_cols])
                df = df[preds == 1]

            except Exception:
                pass

        return df