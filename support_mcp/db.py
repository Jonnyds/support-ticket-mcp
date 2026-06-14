"""Loads the ticket CSV into in-memory DuckDB, cleans it, and provides schema
introspection plus safe read-only SQL execution.

Columns: subject, body, answer, type, queue, priority (low/medium/high),
language (en/de), version, tag_1..tag_8."""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

# config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "tickets.csv"
TABLE = "tickets"
MAX_ROWS = 500  # cap on rows returned by a single query

# categoricals to list (with their distinct values) for the model
CATEGORICAL_COLS = ["type", "queue", "priority", "language"]

# free-text columns (used by search, not SQL)
TEXT_COLS = ["subject", "body", "answer"]

_LOCK = threading.Lock()
_CON: duckdb.DuckDBPyConnection | None = None


# loading & cleaning

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case the headers."""
    return df.rename(
        columns=lambda c: re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower()
    )


def _tag_columns(names) -> list[str]:
    """Find the tag_N columns from a list of column names (or a DataFrame)."""
    cols = names.columns if hasattr(names, "columns") else names
    return sorted([c for c in cols if re.fullmatch(r"tag_?\d+", c)])


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)

    # trim whitespace; empty/"nan" -> NULL
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace(
            {"nan": None, "None": None, "": None}
        )

    # lowercase priority/language for predictable filters
    if "priority" in df.columns:
        df["priority"] = df["priority"].str.lower()
    if "language" in df.columns:
        df["language"] = df["language"].str.lower()

    # join all tags into one column for easy LIKE filters
    tags = _tag_columns(df)
    if tags:
        df["tags_all"] = (
            df[tags].fillna("").agg(
                lambda row: " | ".join([t for t in row if t]), axis=1
            ).replace({"": None})
        )

    return df


def _resolve_csv_path() -> Path:
    """Use TICKETS_CSV if set, otherwise data/tickets.csv."""
    env_path = os.getenv("TICKETS_CSV")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"TICKETS_CSV points to a missing file: {p}")
    if DEFAULT_CSV.exists():
        return DEFAULT_CSV
    raise FileNotFoundError(
        "No ticket data found. Run `python download_data.py` to fetch the dataset, "
        "or place a CSV at data/tickets.csv."
    )


def get_connection() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the cleaned table, loaded once."""
    global _CON
    with _LOCK:
        if _CON is not None:
            return _CON
        csv_path = _resolve_csv_path()
        df = _clean(pd.read_csv(csv_path))
        # block file/network access inside queries (no read_csv, httpfs, etc.)
        con = duckdb.connect(
            database=":memory:", config={"enable_external_access": "false"}
        )
        con.register("df_view", df)
        con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM df_view")
        con.unregister("df_view")
        _CON = con
        return _CON


def get_dataframe() -> pd.DataFrame:
    """Cleaned dataframe (used by search)."""
    return get_connection().execute(f"SELECT * FROM {TABLE}").fetchdf()


# schema introspection

@lru_cache(maxsize=1)
def schema_description() -> dict:
    """Schema the host LLM uses to write SQL."""
    con = get_connection()
    # PRAGMA table_info lists every column and its DuckDB type
    cols = con.execute(f"PRAGMA table_info('{TABLE}')").fetchdf()
    row_count = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    col_names = list(cols["name"].values)
    tag_cols = _tag_columns(col_names)

    columns = []
    for _, r in cols.iterrows():
        name = r["name"]
        entry = {"name": name, "type": str(r["type"])}
        if name in CATEGORICAL_COLS:
            # list the real values so the model uses correct spellings/casing
            vals = con.execute(
                f"SELECT DISTINCT {name} FROM {TABLE} "
                f"WHERE {name} IS NOT NULL ORDER BY 1"
            ).fetchall()
            # DuckDB returns 1-col rows as tuples, e.g. [("high",)]; unwrap to ["high"]
            entry["sample_values"] = [v[0] for v in vals]
        # how many cells are empty — signals column completeness (e.g. tags mostly NULL)
        null_ct = con.execute(
            f'SELECT COUNT(*) FROM {TABLE} WHERE "{name}" IS NULL'
        ).fetchone()[0]
        entry["null_count"] = int(null_ct)
        columns.append(entry)

    # ready-made example query for counting tags across all tag columns;
    # the trailing `or ...` keeps it valid SQL if a CSV has no tag columns
    tag_union = " UNION ALL ".join(
        f"SELECT {t} AS tag FROM {TABLE} WHERE {t} IS NOT NULL" for t in tag_cols
    ) or "SELECT NULL AS tag"

    return {
        "table": TABLE,
        "row_count": int(row_count),
        "columns": columns,
        "text_columns": [c for c in TEXT_COLS if c in col_names],
        "tag_columns": tag_cols,
        "notes": [
            "priority values are lowercase: 'low', 'medium', 'high'.",
            "language is 'en' or 'de' — the dataset is bilingual (English/German).",
            "'body' is the customer's message; 'answer' is the reply/response to "
            "the ticket; both are long free text. Use search_tickets (not SQL LIKE) "
            "for theme questions over them.",
            "type is one of: Incident, Request, Problem, Change.",
            "queue is the routing department (e.g. 'Technical Support', "
            "'Billing and Payments') — see sample_values.",
            "Tags are spread across tag_1..tag_N (many NULL). To count the most "
            "common tags, UNION the tag columns, e.g.: "
            f"SELECT tag, COUNT(*) c FROM ({tag_union}) GROUP BY tag ORDER BY c DESC. "
            "A convenience column 'tags_all' joins them with ' | ' for LIKE filters.",
        ],
    }


# safe SQL execution

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|"
    r"install|load|pragma|set|call|export|import)\b",
    re.IGNORECASE,
)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")

_AUDIT_LOG = DATA_DIR / "query_audit.log"


def _audit(sql: str, ok: bool, rows: int | None, error: str | None) -> None:
    """Log one JSON line per query (timestamp, SQL, outcome)."""
    try:
        import json as _json
        from datetime import datetime, timezone
        _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOG.open("a") as f:
            f.write(_json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "sql": sql,
                "ok": ok,
                "rows": rows,
                "error": error,
            }) + "\n")
    except Exception:
        pass  # auditing must never break query execution


def run_sql(sql: str) -> dict:
    """Run one read-only SELECT and return rows. Guards: single statement,
    SELECT/WITH only, no write/DDL keywords, auto LIMIT cap."""
    cleaned = sql.strip().rstrip(";").strip()

    # blank out string literals so keywords inside them don't trip the guards
    # (e.g. body LIKE '%drop%' must not be read as a DROP statement)
    scannable = _STRING_LITERAL.sub("''", cleaned)

    # a leftover ';' (strings already blanked) means a second statement
    if ";" in scannable:
        return {"error": "Only a single SQL statement is allowed."}

    # must be read-only: a plain SELECT, or a WITH ... SELECT (CTE)
    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return {"error": "Only SELECT queries (optionally a WITH CTE) are allowed."}

    # reject any write/DDL keyword that appears outside a string literal
    if _FORBIDDEN.search(scannable):
        return {"error": "Query contains a disallowed (write/DDL) keyword."}

    # add a LIMIT if the model didn't. Fetch one extra row (MAX_ROWS + 1) so we
    # can tell whether more rows existed, DuckDB still applies the limit in-engine
    # (it never fetches the whole table), and we trim to MAX_ROWS below.
    if not re.search(r"\blimit\b", scannable, re.IGNORECASE):
        cleaned = f"{cleaned}\nLIMIT {MAX_ROWS + 1}"

    try:
        df = get_connection().execute(cleaned).fetchdf()
    except Exception as exc:  # return the error so the model can retry
        _audit(sql, ok=False, rows=None, error=str(exc))
        return {"error": f"SQL error: {exc}"}

    # record truncation before cutting, so the flag is accurate, head() is a
    # backstop for when the model wrote its own LIMIT larger than MAX_ROWS
    truncated = len(df) > MAX_ROWS
    if truncated:
        df = df.head(MAX_ROWS)
    _audit(sql, ok=True, rows=len(df), error=None)

    # make results JSON-safe: NaN -> None (so it serializes to null)
    records = df.astype(object).where(pd.notnull(df), None)

    return {
        "row_count": len(records),
        "columns": list(records.columns),
        "rows": records.to_dict(orient="records"),  # list of {column: value} dicts
        "truncated": truncated,                      # were there more rows than the cap?
    }
