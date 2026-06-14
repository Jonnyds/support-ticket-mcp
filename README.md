# Customer Support Ticket MCP Server

An MCP server that lets you ask natural-language questions about the
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
dataset (bilingual EN/DE support tickets) and get accurate answers —
through **Claude Code** or **Codex**, locally.

## How it works (thin server, host does the reasoning)

The dataset is a mix of **categorical fields** (type, queue, priority, language,
tags) and **free text** (subject, body, answer). Those need different tools:

| Question type | Example | Tool |
|---|---|---|
| Counts / aggregates | "How many tickets per queue?" | `query_tickets` (SQL → DuckDB) |
| Group-bys / breakdowns | "Breakdown by priority and language?" | `query_tickets` |
| Most common tags | "What are the top ticket tags?" | `query_tickets` (UNION tag_1..tag_8) |
| Themes / paraphrase | "What are customers saying about billing?" | `search_tickets` (embeddings) |
| See the data shape | (first call, always) | `get_schema` |

The MCP **host** (Claude Code / Codex) is the chat interface *and* the LLM. It
reads the schema, writes the SQL itself, and picks which tool to call. This
server stays a thin, safe data layer — so the SQL tools need **no API key**.

```
You ─▶ Claude Code (writes SQL, routes) ─stdio─▶ this server ─▶ DuckDB + vector index
```

## Tools

- **`get_schema()`** — columns, types, sample values for categoricals, null counts, usage notes. Call first.
- **`query_tickets(sql)`** — runs a read-only `SELECT` on the `tickets` table (columns: subject, body, answer, type, queue, priority, language, version, tag_1..tag_8, tags_all). Guards: single statement, SELECT/WITH only, no DDL/DML, SQL results capped at 500 rows, errors returned to the model for self-correction.
- **`search_tickets(query, k=5)`** — semantic search over subject+body via OpenAI embeddings (EN/DE). Returns up to `k` results (capped at 20), filtered by a relevance floor so off-topic queries return nothing. **Needs `OPENAI_API_KEY`.**

## Requirements

- Python 3.10+
- Packages in `requirements.txt` (`mcp[cli]`, `duckdb`, `pandas`, `numpy`, `openai`, `python-dotenv`)

## Setup

```bash
git clone <your-repo-url>
cd support-ticket-mcp

# create a virtualenv (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell
# .venv\Scripts\activate.bat       # Windows cmd

pip install -r requirements.txt

# 1) Download the dataset -> data/tickets.csv
python download_data.py

# 2) Only needed for semantic search: add your OpenAI key
cp .env.example .env        # Windows: copy .env.example .env  — then edit it

# 3) (recommended) prebuild the embedding index so the first search is instant
#    — otherwise it builds lazily on the first search_tickets call:
python -m support_mcp.build_index
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | only for `search_tickets` | embeddings for semantic search |
| `OPENAI_EMBED_MODEL` | no | embedding model, defaults to `text-embedding-3-small` |
| `SEARCH_MIN_SIMILARITY` | no | relevance floor for search (0–1), defaults to `0.30` |
| `TICKETS_CSV` | no | point the server at a custom CSV path |

(The download source is set in `download_data.py`; override it with a
`DATASET_URL` environment variable when running that script if needed.)

## Run it standalone (sanity check)

```bash
python support_mcp/server.py        # starts on stdio; Ctrl-C to stop
```

Optionally inspect the tools without an LLM using the MCP Inspector (an official
local debug UI for MCP servers):

```bash
npx @modelcontextprotocol/inspector python support_mcp/server.py
```

## Connect to Claude Code

From the project directory, register the server (this is the form that works —
note the `--` before the command):

```bash
claude mcp add support-tickets -- python support_mcp/server.py
```

Then launch the interface and start asking:

```bash
claude
```

If the server can't find its packages, your virtualenv may not be active when
Claude Code launches it — point the command at the venv's Python directly, e.g.
`.../.venv/bin/python support_mcp/server.py` (or `...\.venv\Scripts\python.exe` on Windows).

## Connect to Codex

The server is host-agnostic. It was developed and tested with Claude Code; to use
it from Codex, add it to `~/.codex/config.toml` with its standard MCP config and
run `codex` from the project directory:

```toml
[mcp_servers.support-tickets]
command = "python"
args = ["support_mcp/server.py"]
cwd = "/absolute/path/to/support-ticket-mcp"
env = { OPENAI_API_KEY = "sk-..." }
```

## Example questions to try

- "How many tickets are there per queue?"
- "What's the breakdown by priority?"
- "Show the split between English and German tickets."
- "What are the 10 most common tags?"
- "How many tickets are type Incident with high priority?"
- "Find tickets about login or VPN problems." *(uses search_tickets)*

## Notes / data quirks handled on load

- `priority` is normalized to lowercase (`low`/`medium`/`high`) for predictable filters.
- The dataset is **bilingual (en/de)** — `language` lets you filter or split by it.
- `body` = the customer's message text, `answer` = the reply/response to the ticket.
- Tags span `tag_1..tag_8` (often NULL); a `tags_all` column joins them for easy `LIKE` filters.

## Safety & observability

- **Engine-level sandbox:** DuckDB runs with `enable_external_access=false`, so queries cannot read files or the network.
- **Query guards:** single statement, SELECT/WITH only, DDL/DML blocked (string literals stripped before scanning, so text like `'%set up%'` is fine), 500-row cap.
- **Audit log + stderr logging:** every executed query is appended as a JSON line to `data/query_audit.log` (timestamp, SQL, outcome, row count); tool calls are also logged to stderr, which the host displays.