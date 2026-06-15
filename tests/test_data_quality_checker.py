"""Unit tests for the Automated Data Quality Checker."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_quality_checker import DataQualityChecker


SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_data.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_checker_from_df(df: pd.DataFrame, tmp_path: Path) -> DataQualityChecker:
    """Write a DataFrame to a temp CSV and return a DataQualityChecker for it."""
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    return DataQualityChecker(str(p))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoadFile:
    def test_load_csv_shape(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        checker = make_checker_from_df(df, tmp_path)
        loaded = checker.load_file()
        assert loaded.shape == (2, 2)

    def test_load_csv_normalises_missing_markers(self, tmp_path):
        df = pd.DataFrame({"col": ["NA", "N/A", "null", "", "valid"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        assert checker.df["col"].isna().sum() == 4

    def test_load_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("hello")
        checker = DataQualityChecker(str(p))
        with pytest.raises(ValueError, match="Unsupported file type"):
            checker.load_file()

    def test_load_sample_csv(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.load_file()
        assert checker.df.shape[0] == 15
        assert "name" in checker.df.columns


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

class TestMissingValues:
    def test_missing_values_detected(self, tmp_path):
        df = pd.DataFrame({"a": ["1", None, "3"], "b": ["x", "y", "z"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_missing_values()
        assert len(issues) == 1
        assert issues[0]["column"] == "a"
        assert issues[0]["missing_count"] == 1
        assert issues[0]["missing_pct"] == pytest.approx(33.33, abs=0.01)

    def test_no_missing_values(self, tmp_path):
        df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        assert checker.check_missing_values() == []

    def test_row_numbers_are_one_indexed_with_header(self, tmp_path):
        df = pd.DataFrame({"a": [None, "2", None]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_missing_values()
        # Row 0 → file line 2, row 2 → file line 4
        assert 2 in issues[0]["rows"]
        assert 4 in issues[0]["rows"]

    def test_sample_csv_missing_values(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.run_all_checks()
        affected_cols = [i["column"] for i in checker.issues["missing_values"]]
        assert "age" in affected_cols
        assert "email" in affected_cols


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_duplicates_detected(self, tmp_path):
        df = pd.DataFrame({"a": ["1", "1", "2"], "b": ["x", "x", "y"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_duplicates()
        assert len(issues) == 1
        assert issues[0]["count"] == 2

    def test_no_duplicates(self, tmp_path):
        df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        assert checker.check_duplicates() == []

    def test_duplicate_row_numbers(self, tmp_path):
        df = pd.DataFrame({"a": ["same", "same", "diff"], "b": ["v", "v", "w"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_duplicates()
        # Rows 0,1 → file lines 2,3
        assert 2 in issues[0]["duplicate_rows"]
        assert 3 in issues[0]["duplicate_rows"]

    def test_sample_csv_has_duplicate(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.run_all_checks()
        assert len(checker.issues["duplicates"]) >= 1
        # Alice Smith appears twice (rows 1 and 11 → file lines 2 and 12)
        all_rows = [r for item in checker.issues["duplicates"] for r in item["duplicate_rows"]]
        assert 2 in all_rows
        assert 12 in all_rows


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

class TestOutliers:
    def test_outlier_detected(self, tmp_path):
        values = ["10", "11", "10", "12", "11", "10", "9", "1000"]
        df = pd.DataFrame({"score": values})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_outliers()
        assert len(issues) == 1
        assert issues[0]["column"] == "score"
        assert 1000.0 in issues[0]["values"]

    def test_no_outliers_in_normal_data(self, tmp_path):
        values = [str(i) for i in range(10, 21)]
        df = pd.DataFrame({"score": values})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        assert checker.check_outliers() == []

    def test_non_numeric_column_skipped(self, tmp_path):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "Diana", "Eve"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        assert checker.check_outliers() == []

    def test_sample_csv_outliers(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.run_all_checks()
        outlier_cols = [i["column"] for i in checker.issues["outliers"]]
        # age=200 and salary=999999 are extreme
        assert "age" in outlier_cols or "salary" in outlier_cols


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_whitespace_detected(self, tmp_path):
        df = pd.DataFrame({"col": [" leading", "normal", "trailing "]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_formatting()
        types = [i["type"] for i in issues if i["column"] == "col"]
        assert "leading_trailing_whitespace" in types

    def test_no_whitespace_issue(self, tmp_path):
        df = pd.DataFrame({"col": ["clean", "values", "here"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = [i for i in checker.check_formatting() if i["type"] == "leading_trailing_whitespace"]
        assert issues == []

    def test_case_inconsistency_detected(self, tmp_path):
        df = pd.DataFrame({"status": ["active", "Active", "ACTIVE", "inactive"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_formatting()
        types = [i["type"] for i in issues if i["column"] == "status"]
        assert "case_inconsistency" in types

    def test_no_case_inconsistency(self, tmp_path):
        df = pd.DataFrame({"status": ["active", "inactive", "pending"]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = [i for i in checker.check_formatting() if i["type"] == "case_inconsistency"]
        assert issues == []

    def test_mixed_date_formats_detected(self, tmp_path):
        df = pd.DataFrame({"date": [
            "2023-01-15", "2023-02-20", "03/10/2023", "04/05/2023",
            "2023-05-01", "2023-06-10",
        ]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_formatting()
        types = [i["type"] for i in issues if i["column"] == "date"]
        assert "mixed_date_formats" in types

    def test_invalid_email_detected(self, tmp_path):
        df = pd.DataFrame({"email": [
            "valid@example.com", "also.valid@test.org",
            "not-an-email", "missing@tld",
        ]})
        checker = make_checker_from_df(df, tmp_path)
        checker.load_file()
        issues = checker.check_formatting()
        types = [i["type"] for i in issues if i["column"] == "email"]
        assert "invalid_email_format" in types

    def test_sample_csv_formatting_issues(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.run_all_checks()
        all_types = [i["type"] for i in checker.issues["formatting"]]
        # Expect at least case and email/date issues in the sample data
        assert len(checker.issues["formatting"]) >= 2
        assert any(t in all_types for t in (
            "case_inconsistency", "mixed_date_formats", "invalid_email_format",
            "leading_trailing_whitespace",
        ))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_text_report_contains_sections(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.run_all_checks()
        report = checker.generate_report("text")
        assert "MISSING VALUES" in report
        assert "DUPLICATE ROWS" in report
        assert "OUTLIERS" in report
        assert "FORMATTING INCONSISTENCIES" in report
        assert "TOTAL ISSUES FOUND" in report

    def test_json_report_is_valid(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.run_all_checks()
        raw = checker.generate_report("json")
        data = json.loads(raw)
        assert "summary" in data
        assert "issues" in data
        assert set(data["issues"].keys()) == {"missing_values", "duplicates", "outliers", "formatting"}

    def test_json_summary_fields(self):
        checker = DataQualityChecker(str(SAMPLE_CSV))
        checker.run_all_checks()
        data = json.loads(checker.generate_report("json"))
        summary = data["summary"]
        assert summary["rows"] == 15
        assert "checked_at" in summary


# ---------------------------------------------------------------------------
# CLI exit code
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_exits_1_on_issues(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "data_quality_checker.py"),
             str(SAMPLE_CSV)],
            capture_output=True, text=True,
        )
        assert result.returncode == 1

    def test_cli_exits_0_on_clean_data(self, tmp_path):
        df = pd.DataFrame({"id": ["1", "2"], "name": ["Alice", "Bob"], "score": ["80", "90"]})
        p = tmp_path / "clean.csv"
        df.to_csv(p, index=False)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "data_quality_checker.py"), str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_exits_2_on_bad_file(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "data_quality_checker.py"),
             str(tmp_path / "nonexistent.csv")],
            capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_cli_json_flag(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "data_quality_checker.py"),
             str(SAMPLE_CSV), "--format", "json"],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        assert "summary" in data

    def test_cli_output_file(self, tmp_path):
        out = tmp_path / "report.txt"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "data_quality_checker.py"),
             str(SAMPLE_CSV), "--output", str(out)],
            capture_output=True, text=True,
        )
        assert out.exists()
        content = out.read_text()
        assert "TOTAL ISSUES FOUND" in content
