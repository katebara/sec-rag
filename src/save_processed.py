"""
Run the parser and save structured output to data/processed/ as
one JSON file per filing.
"""

import json
from pathlib import Path
from src.parse import process_all, OUT_DIR

def save_results(results):
    for r in results:
        filename = f"{r['company']}_{r['fiscal_year']}.json"
        outpath = OUT_DIR / filename
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"Saved {outpath}")

if __name__ == "__main__":
    results = process_all()
    save_results(results)
    print(f"\nSaved {len(results)} processed filings to {OUT_DIR}")
