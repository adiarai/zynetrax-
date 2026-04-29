import os
from intelligence.row_detector import RowDetector


class TextProcessor:

    @staticmethod
    def process_text_file(file_path):
        """
        Process raw text file and convert it into a structured DataFrame.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError("Text file not found.")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Step 1: Detect row boundaries
        row_result = RowDetector.detect_rows(text)

        if row_result["confidence"] < 0.5:
            raise ValueError("Insufficient structured repetition detected in text.")

        # Step 2: Extract structured fields
        df = RowDetector.extract_fields(row_result["rows"])

        if df.empty:
            raise ValueError("Failed to extract structured data from text.")

        return df, row_result