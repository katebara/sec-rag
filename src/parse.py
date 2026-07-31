"""
Parse raw SEC 10-K filings (full-submission.txt) into clean,
section-labeled text chunks.
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw/sec-edgar-filings")
OUT_DIR = Path("data/processed")

# Case-insensitive, handles "Item"/"ITEM", regular or non-breaking spaces,
# and titles that appear on the same line OR the next line.
ITEM_PATTERN = re.compile(
    r'Item\s+(\d{1,2}[A-C]?)\.\s*\n?\s*([A-Z][A-Za-z,\u2019\'\-\s]{2,80})',
    re.IGNORECASE
)

CROSS_REF_TRIGGERS = ("refer to", "see ", "pursuant to", "under item", "described in")

def extract_html_from_submission(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    match = re.search(r"<TEXT>(.*?)</TEXT>", content, re.DOTALL)
    return match.group(1) if match else ""

def html_to_clean_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")  # normalize non-breaking spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def is_toc_entry(text, match):
    after = text[match.end():match.end() + 60]
    return bool(re.search(r'\n\s*\d{1,3}\s*\n', after))

def is_cross_reference(text, match):
    before = text[max(0, match.start() - 40):match.start()].lower()
    if '"' in before or "\u201c" in before:
        return True
    return any(trigger in before for trigger in CROSS_REF_TRIGGERS)

def split_by_item(text):
    raw_matches = list(ITEM_PATTERN.finditer(text))
    real_matches = [
        m for m in raw_matches
        if not is_toc_entry(text, m) and not is_cross_reference(text, m)
    ]

    seen = set()
    filtered = []
    for m in real_matches:
        item_num = m.group(1).upper()
        if item_num not in seen:
            seen.add(item_num)
            filtered.append(m)

    sections = {}
    for i, m in enumerate(filtered):
        start = m.start()
        end = filtered[i + 1].start() if i + 1 < len(filtered) else len(text)
        header = f"Item {m.group(1).upper()}. {m.group(2).strip()}"
        sections[header] = text[start:end].strip()
    return sections

def process_filing(filepath, company, fiscal_year):
    html = extract_html_from_submission(filepath)
    if not html:
        print(f"  WARNING: no <TEXT> block found in {filepath}")
        return None
    text = html_to_clean_text(html)
    sections = split_by_item(text)
    return {
        "company": company,
        "fiscal_year": fiscal_year,
        "sections": sections,
        "num_sections_found": len(sections),
    }

def process_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for company_dir in RAW_DIR.iterdir():
        if not company_dir.is_dir():
            continue
        company = company_dir.name
        tenk_dir = company_dir / "10-K"
        if not tenk_dir.exists():
            continue
        for filing_dir in tenk_dir.iterdir():
            filepath = filing_dir / "full-submission.txt"
            if not filepath.exists():
                continue
            print(f"Processing {company} — {filing_dir.name}")
            result = process_filing(filepath, company, filing_dir.name)
            if result:
                print(f"  Found {result['num_sections_found']} Item sections")
                results.append(result)
    return results

if __name__ == "__main__":
    results = process_all()
    print(f"\nProcessed {len(results)} filings total.")
