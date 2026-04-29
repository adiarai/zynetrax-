import json


class ExportManager:

    @staticmethod
    def export(df, report, output_prefix="output"):

        clean_path = f"{output_prefix}_clean.csv"
        report_path = f"{output_prefix}_report.json"

        df.to_csv(clean_path, index=False)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4, default=str)

        return {
            "clean_file": clean_path,
            "report_file": report_path
        }