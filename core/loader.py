import pandas as pd
import chardet
import os


class FileLoader:

    @staticmethod
    def detect_encoding(file_path):
        with open(file_path, "rb") as f:
            raw_data = f.read(100000)
        return chardet.detect(raw_data)["encoding"]

    @staticmethod
    def load_file(file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            encoding = FileLoader.detect_encoding(file_path)
            return pd.read_csv(file_path, encoding=encoding)

        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        elif ext == ".json":
            return pd.read_json(file_path)

        else:
            raise ValueError("Unsupported file type")