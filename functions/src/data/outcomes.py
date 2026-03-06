import os
import pandas as pd


def load_outcomes(
    path: str,
    *,
    hex_col: str = "hex",
    year_col: str = "year",
    value_col: str = "value",
) -> pd.DataFrame:
    """Load an outcome table keyed by (hex, year).

    Expected columns are (hex_col, year_col, value_col). The returned DataFrame
    is standardized to columns: hex, year, value.

    Supported formats: .csv, .parquet
    """

    if not path:
        raise ValueError("outcomes path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Outcomes file not found: {path}")

    lower = path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported outcomes format. Use .csv or .parquet")

    missing = [c for c in (hex_col, year_col, value_col) if c not in df.columns]
    if missing:
        raise ValueError(
            "Outcomes file missing required columns: "
            + ", ".join(missing)
            + f". Present columns: {', '.join(df.columns)}"
        )

    out = df[[hex_col, year_col, value_col]].copy()
    out = out.rename(columns={hex_col: "hex", year_col: "year", value_col: "value"})
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["hex", "year", "value"]).copy()
    out["year"] = out["year"].astype(int)

    return out
