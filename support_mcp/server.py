"""MCP server (stdio) with three tools for querying the support-ticket dataset.
The host LLM writes the SQL; this server just runs it and does semantic search."""

from __future__ import annotations

import json
import logging
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# log to stderr (the MCP host surfaces it); never stdout, which carries the protocol
logging.basicConfig(
    level=logging.INFO, stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("support-tickets")

# import whether launched as a module or by file path
try:
    from . import db, search
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from support_mcp import db, search

mcp = FastMCP("support-tickets")


@mcp.tool()
def get_schema() -> str:
    """Return the tickets table schema, column types, sample values, and notes.
    Call this before writing SQL so you can use the correct column names and
    exact categorical values, such as priority, queue, type, language, and tags."""
    log.info("get_schema called")
    return json.dumps(db.schema_description(), indent=2, default=str)


@mcp.tool()
def query_tickets(sql: str) -> str:
    """Run a safe read-only DuckDB SQL query on the 'tickets' table.
    Use this for exact structured questions: counts, filters, group-bys,
    comparisons, and aggregations by queue, priority, type, language, or tags.
    Only SELECT/WITH statements are allowed. Results are capped at 500 rows.
    Call get_schema first to confirm column names and exact field values.
    Do not use this for natural-language ticket content, use search_tickets
    instead.
    """
    log.info("query_tickets: %s", sql.replace("\n", " ")[:200])
    return json.dumps(db.run_sql(sql), indent=2, default=str)


@mcp.tool()
def search_tickets(query: str, k: int = 5) -> str:
    """Run semantic search over ticket subject and body text.
    Use this for natural-language content questions such as billing complaints,
    login problems, refund issues, angry customers, or similar wording.
    Returns the top-k most similar tickets, with similarity scores and ticket
    fields. Use query_tickets for exact counts, filters, and aggregations.
    """
    try:
        log.info("search_tickets: %r (k=%s)", query[:100], k)
        return json.dumps(search.search(query, k), indent=2, default=str)
    except Exception as exc:
        log.warning("search_tickets failed: %s", exc)
        return json.dumps({"error": str(exc)})


def main() -> None:
    try:
        log.info("loading data...")
        con = db.get_connection()
        rows = con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        log.info("ready: %s tickets loaded; starting server", rows)
    except FileNotFoundError as exc:
        raise SystemExit(f"Startup failed: {exc}\nRun `python download_data.py` first.")
    mcp.run()


if __name__ == "__main__":
    main()
