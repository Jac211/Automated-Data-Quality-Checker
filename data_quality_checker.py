#!/usr/bin/env python3
"""
Automated Data Quality Checker
Ingests CSV/Excel files and flags missing values, duplicates, outliers,
and formatting inconsistencies.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


MISSING_MARKERS = {"", "NA", "N/A", "n/a", "null", "NULL", "None", "NONE", "nan", "NaN", "-", "#N/A"}

DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
    "%d-%m-%Y", "%m-%d-%Y", "%Y%m%d",
    "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
    "%b %d, %Y", "%B %d, %Y",
]

EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')


def _try_parse_date(value: str) -> Optional[str]:
    """Return matched format string if value parses as a date, else None."""
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(value.strip(), fmt)
            return fmt
        except ValueError:
            continue
    return None


class DataQualityChecker:
    """Checks data quality for CSV and Excel files."""

    def __init__(self, filepath: str, sheet_name: Optional[str] = None):
        self.filepath = Path(filepath)
        self.sheet_name = sheet_name
        self.df: Optional[pd.DataFrame] = None
        self.issues: dict = {
            "missing_values": [],
            "duplicates": [],
            "outliers": [],
            "formatting": [],
        }
        self.summary: dict = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_file(self) -> pd.DataFrame:
        """Load a CSV or Excel file into a DataFrame, normalizing missing markers."""
        suffix = self.filepath.suffix.lower()
        if suffix == ".csv":
            self.df = pd.read_csv(self.filepath, dtype=str, keep_default_na=False)
        elif suffix in (".xlsx", ".xls"):
            kwargs: dict = {}
            if self.sheet_name:
                kwargs["sheet_name"] = self.sheet_name
            self.df = pd.read_excel(self.filepath, dtype=str, keep_default_na=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Supported: .csv, .xlsx, .xls")

        self.df.replace(list(MISSING_MARKERS), pd.NA, inplace=True)
        return self.df

    # ------------------------------------------------------------------
    # Check 1: Missing values
    # ------------------------------------------------------------------

    def check_missing_values(self) -> list:
        """Flag columns that contain missing/null values."""
        issues = []
        for col in self.df.columns:
            mask = self.df[col].isna()
            if mask.any():
                rows = [i + 2 for i in self.df.index[mask].tolist()]  # +2: 1-indexed + header row
                issues.append({
                    "column": col,
                    "missing_count": int(mask.sum()),
                    "missing_pct": round(float(mask.mean()) * 100, 2),
                    "rows": rows,
                })
        self.issues["missing_values"] = issues
        return issues

    # ------------------------------------------------------------------
    # Check 2: Duplicate rows
    # ------------------------------------------------------------------

    def check_duplicates(self) -> list:
        """Flag groups of fully-identical rows."""
        issues = []
        dup_mask = self.df.duplicated(keep=False)
        if not dup_mask.any():
            self.issues["duplicates"] = issues
            return issues

        # Group duplicates by content hash to avoid NA-groupby issues
        groups: dict = {}
        for idx in self.df.index[dup_mask].tolist():
            key = tuple(str(v) for v in self.df.loc[idx])
            groups.setdefault(key, []).append(idx)

        for key, indices in groups.items():
            sample = {col: self.df.loc[indices[0], col] for col in list(self.df.columns)[:3]}
            issues.append({
                "duplicate_rows": [i + 2 for i in indices],
                "count": len(indices),
                "sample_values": sample,
            })

        self.issues["duplicates"] = issues
        return issues

    # ------------------------------------------------------------------
    # Check 3: Outliers
    # ------------------------------------------------------------------

    def check_outliers(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5) -> list:
        """Flag outliers in numeric columns using both IQR and Z-score methods."""
        issues = []
        for col in self.df.columns:
            series = pd.to_numeric(self.df[col], errors="coerce")
            valid = series.dropna()

            # Skip columns that are not predominantly numeric
            if len(valid) < 4 or series.isna().mean() > 0.5:
                continue

            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            mean = float(valid.mean())
            std = float(valid.std())

            # Skip constant columns
            if iqr == 0 and std == 0:
                continue

            iqr_lower = q1 - iqr_multiplier * iqr
            iqr_upper = q3 + iqr_multiplier * iqr

            iqr_mask = (series < iqr_lower) | (series > iqr_upper)
            z_mask = (series - mean).abs() / (std if std > 0 else 1) > z_threshold
            combined = (iqr_mask | z_mask) & series.notna()

            if combined.any():
                outlier_indices = self.df.index[combined].tolist()
                issues.append({
                    "column": col,
                    "outlier_count": int(combined.sum()),
                    "rows": [i + 2 for i in outlier_indices],
                    "values": [round(float(series[i]), 4) for i in outlier_indices],
                    "stats": {
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "q1": round(q1, 4),
                        "q3": round(q3, 4),
                        "iqr_bounds": [round(iqr_lower, 4), round(iqr_upper, 4)],
                    },
                })

        self.issues["outliers"] = issues
        return issues

    # ------------------------------------------------------------------
    # Check 4: Formatting inconsistencies
    # ------------------------------------------------------------------

    def check_formatting(self) -> list:
        """Detect formatting inconsistencies across all columns."""
        issues = []
        for col in self.df.columns:
            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                continue
            issues.extend(self._check_whitespace(col, non_null))
            issues.extend(self._check_case_inconsistency(col, non_null))
            issues.extend(self._check_date_format_inconsistency(col, non_null))
            issues.extend(self._check_email_format(col, non_null))
            issues.extend(self._check_numeric_stored_as_string(col, non_null))
        self.issues["formatting"] = issues
        return issues

    def _check_whitespace(self, col: str, series: pd.Series) -> list:
        str_series = series.astype(str)
        affected = str_series[str_series != str_series.str.strip()]
        if affected.empty:
            return []
        return [{
            "column": col,
            "type": "leading_trailing_whitespace",
            "affected_count": len(affected),
            "rows": [i + 2 for i in affected.index.tolist()],
            "examples": affected.head(5).tolist(),
        }]

    def _check_case_inconsistency(self, col: str, series: pd.Series) -> list:
        unique_vals = [str(v) for v in series.unique()]
        # Only check low-cardinality columns that look like categories
        if len(unique_vals) > 30 or len(unique_vals) < 2:
            return []

        case_groups: dict = {}
        for v in unique_vals:
            case_groups.setdefault(v.lower(), []).append(v)

        conflicts = {k: v for k, v in case_groups.items() if len(v) > 1}
        if not conflicts:
            return []
        return [{
            "column": col,
            "type": "case_inconsistency",
            "affected_count": sum(len(v) for v in conflicts.values()),
            "details": f"Same value in different cases: {conflicts}",
        }]

    def _check_date_format_inconsistency(self, col: str, series: pd.Series) -> list:
        format_counts: dict = {}
        non_date = 0

        for val in series.astype(str):
            fmt = _try_parse_date(val)
            if fmt:
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
            else:
                non_date += 1

        total = len(series)
        date_total = sum(format_counts.values())

        if date_total / total >= 0.5 and len(format_counts) > 1:
            return [{
                "column": col,
                "type": "mixed_date_formats",
                "details": f"Multiple date formats detected: {list(format_counts.keys())}",
                "format_counts": format_counts,
            }]
        return []

    def _check_email_format(self, col: str, series: pd.Series) -> list:
        str_series = series.astype(str)
        # Only inspect columns where most values look like emails
        if str_series.str.contains("@", na=False).mean() < 0.5:
            return []

        invalid = str_series[~str_series.apply(lambda x: bool(EMAIL_PATTERN.match(x.strip())))]
        if invalid.empty:
            return []
        return [{
            "column": col,
            "type": "invalid_email_format",
            "affected_count": len(invalid),
            "rows": [i + 2 for i in invalid.index.tolist()],
            "examples": invalid.head(5).tolist(),
        }]

    def _check_numeric_stored_as_string(self, col: str, series: pd.Series) -> list:
        str_series = series.astype(str)
        # Only inspect columns that look predominantly numeric
        numeric_like = str_series.str.match(r'^\$?[\d,]+\.?\d*$').mean()
        if numeric_like < 0.3:
            return []

        with_commas = int(str_series.str.match(r'^\$?[\d]{1,3}(,\d{3})+(\.\d+)?$').sum())
        without_commas = int(str_series.str.match(r'^\d{4,}$').sum())
        if with_commas > 0 and without_commas > 0:
            return [{
                "column": col,
                "type": "inconsistent_number_format",
                "details": f"{with_commas} value(s) with thousand-separator commas, {without_commas} without",
                "affected_count": with_commas + without_commas,
            }]
        return []

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ) -> dict:
        """Load the file and run all four quality checks."""
        self.load_file()
        self.check_missing_values()
        self.check_duplicates()
        self.check_outliers(z_threshold=z_threshold, iqr_multiplier=iqr_multiplier)
        self.check_formatting()

        self.summary = {
            "file": str(self.filepath),
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "missing_value_columns": len(self.issues["missing_values"]),
            "duplicate_groups": len(self.issues["duplicates"]),
            "outlier_columns": len(self.issues["outliers"]),
            "formatting_issues": len(self.issues["formatting"]),
        }
        return {"summary": self.summary, "issues": self.issues}

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_format: str = "text") -> str:
        """Generate a report in 'text' or 'json' format."""
        if output_format == "json":
            return json.dumps({"summary": self.summary, "issues": self.issues}, indent=2, default=str)

        lines = []
        sep = "=" * 70
        thin = "─" * 70

        lines += [
            sep,
            "         AUTOMATED DATA QUALITY REPORT",
            sep,
            f"File    : {self.summary.get('file', 'N/A')}",
            f"Rows    : {self.summary.get('rows', 0):,}",
            f"Columns : {self.summary.get('columns', 0)}",
            f"Checked : {self.summary.get('checked_at', '')}",
            "",
        ]

        # Missing values
        mv = self.issues["missing_values"]
        lines += [thin, f"[1] MISSING VALUES — {len(mv)} column(s) affected", thin]
        if mv:
            for item in mv:
                rows_preview = item["rows"][:10]
                suffix = f" … +{len(item['rows']) - 10} more" if len(item["rows"]) > 10 else ""
                lines += [
                    f"  Column : {item['column']}",
                    f"    Missing : {item['missing_count']} ({item['missing_pct']}%)",
                    f"    Rows    : {rows_preview}{suffix}",
                ]
        else:
            lines.append("  No missing values detected.")
        lines.append("")

        # Duplicates
        dups = self.issues["duplicates"]
        lines += [thin, f"[2] DUPLICATE ROWS — {len(dups)} duplicate group(s)", thin]
        if dups:
            for i, item in enumerate(dups, 1):
                sample = ", ".join(f"{k}={v}" for k, v in item["sample_values"].items())
                lines += [
                    f"  Group {i}: {item['count']} identical rows at file lines {item['duplicate_rows']}",
                    f"    Sample  : {sample}",
                ]
        else:
            lines.append("  No duplicate rows detected.")
        lines.append("")

        # Outliers
        outs = self.issues["outliers"]
        lines += [thin, f"[3] OUTLIERS — {len(outs)} column(s) affected", thin]
        if outs:
            for item in outs:
                s = item["stats"]
                rows_preview = item["rows"][:5]
                vals_preview = [round(v, 2) for v in item["values"][:5]]
                suffix = f" … +{len(item['rows']) - 5} more" if len(item["rows"]) > 5 else ""
                lines += [
                    f"  Column : {item['column']}",
                    f"    Outliers : {item['outlier_count']}",
                    f"    Stats    : mean={s['mean']}, std={s['std']}, "
                    f"IQR bounds=[{s['iqr_bounds'][0]}, {s['iqr_bounds'][1]}]",
                    f"    Rows     : {rows_preview}{suffix}",
                    f"    Values   : {vals_preview}",
                ]
        else:
            lines.append("  No outliers detected.")
        lines.append("")

        # Formatting
        fmt = self.issues["formatting"]
        lines += [thin, f"[4] FORMATTING INCONSISTENCIES — {len(fmt)} issue(s)", thin]
        if fmt:
            for item in fmt:
                issue_label = item["type"].replace("_", " ").title()
                lines.append(f"  Column : {item['column']}  |  Issue : {issue_label}")
                if "details" in item:
                    lines.append(f"    {item['details']}")
                if "affected_count" in item:
                    lines.append(f"    Affected : {item['affected_count']} value(s)")
                if "rows" in item:
                    rows_preview = item["rows"][:5]
                    suffix = f" … +{len(item['rows']) - 5} more" if len(item["rows"]) > 5 else ""
                    lines.append(f"    Rows     : {rows_preview}{suffix}")
                if "examples" in item:
                    lines.append(f"    Examples : {item['examples'][:5]}")
        else:
            lines.append("  No formatting inconsistencies detected.")
        lines.append("")

        # Summary footer
        total = len(mv) + len(dups) + len(outs) + len(fmt)
        lines += [sep, f"  TOTAL ISSUES FOUND: {total}", sep]

        return "\n".join(lines)

    def has_issues(self) -> bool:
        return any(self.issues[k] for k in self.issues)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_quality_checker",
        description="Automated Data Quality Checker — flags missing values, duplicates, "
                    "outliers, and formatting inconsistencies in CSV/Excel files.",
    )
    parser.add_argument("file", help="Path to the CSV or Excel file to check")
    parser.add_argument("--sheet", default=None, help="Sheet name (Excel only)")
    parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text",
        help="Report output format (default: text)",
    )
    parser.add_argument("--output", default=None, help="Write report to this file instead of stdout")
    parser.add_argument(
        "--z-threshold", type=float, default=3.0,
        help="Z-score threshold for outlier detection (default: 3.0)",
    )
    parser.add_argument(
        "--iqr-multiplier", type=float, default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checker = DataQualityChecker(args.file, sheet_name=args.sheet)
    try:
        checker.run_all_checks(z_threshold=args.z_threshold, iqr_multiplier=args.iqr_multiplier)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    report = checker.generate_report(output_format=args.output_format)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 1 if checker.has_issues() else 0


if __name__ == "__main__":
    sys.exit(main())
