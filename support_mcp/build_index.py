"""Prebuild and cache the embedding index so the first search is instant.
Usage: python -m support_mcp.build_index"""

import time

from dotenv import load_dotenv

load_dotenv()

from . import search  # noqa: E402


def main() -> None:
    print("Building embedding index (one-time, cached afterwards)...")
    print("Embedding ticket text via OpenAI — this can take a minute on first run.")
    start = time.perf_counter()
    try:
        search._ensure_index()
    except RuntimeError as exc:
        # raised when OPENAI_API_KEY is missing
        raise SystemExit(f"Cannot build index: {exc}")
    except FileNotFoundError as exc:
        # raised when there is no ticket CSV yet
        raise SystemExit(f"Cannot build index: {exc}\nRun `python download_data.py` first.")
    except Exception as exc:
        raise SystemExit(f"Index build failed: {exc}")

    n = len(search._META) if search._META else 0
    elapsed = time.perf_counter() - start
    print(f"Done. Indexed {n:,} tickets in {elapsed:.1f}s. Cache: data/.cache/")


if __name__ == "__main__":
    main()
