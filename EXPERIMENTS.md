# Experiments Log

Tracking decisions, comparisons, and results as the SEC 10-K RAG pipeline is built.

---

## Experiment: Section-aware chunking regex — initial version
**Date:** July 2026

- First pass split filings on a simple `Item N.` regex.
- Result: massive over-matching — 40+ "sections" found per filing instead of the expected ~15.
- Cause: regex matched every occurrence of "Item N." anywhere in the text, including:
  - Table of Contents entries (each item listed once with a page number)
  - In-text cross-references (e.g. "Refer to Item 1A. Risk Factors")

**Fix:** added two filters before accepting a match as a real header:
- `is_toc_entry`: rejects matches immediately followed by a standalone page number
- `is_cross_reference`: rejects matches preceded by phrases like "refer to", "see", or a quotation mark

Result after fix: NVDA/QCOM dropped to a reasonable ~18–21 sections. AMD returned **0** sections, AVGO returned only **2**.

---

## Experiment: Cross-filer header format differences
**Date:** July 2026

Debugged AMD's raw text directly (via `repr()` to reveal hidden formatting) and found:
- AMD uses all-caps `ITEM` (not `Item`) with a non-breaking space (`\xa0`) before the number
- The section title falls on the line *after* the header, not the same line

**Fix:**
- Made the regex case-insensitive
- Allowed an optional newline between the item number and the title
- Normalized non-breaking spaces to regular spaces during text cleanup

**Result:** AMD went from 0 → 17–19 sections. AVGO went from 2 → 20–21 sections. All four "normal" filers (NVDA, AMD, QCOM, AVGO) landed in the expected ~15 standard items + legitimate sub-items (1A/1B/1C, 7/7A) range.

| Filer | Before fix | After fix |
|---|---|---|
| NVDA | 44–47 (over-matched) | 19–21 |
| AMD  | 0 | 17–19 |
| QCOM | 45–47 (over-matched) | 18–20 |
| AVGO | 2 | 20–21 |

---

## Experiment: INTC still under-matching after cross-filer fix
**Date:** July 2026

INTC consistently returned only 8–10 sections, even after the AMD-style fixes. Investigation:

1. All matched "Item N." occurrences traced back to a single **Cross-Reference Index** page near the front of the filing — a table using page *ranges* (e.g. "Pages 31-46") instead of single page numbers, which our TOC filter wasn't designed to catch.
2. Searched for the real narrative content ("General development of business") — found it only appears **once**, at 96–99% through the cleaned document, which is abnormal.
3. Hypothesized hidden XBRL fact blocks (`style="display:none"`, used to embed thousands of invisible tagged financial values) were bloating the document and pushing real content to the end. Added a strip step for these (2,413 elements removed, doc length dropped ~527k → ~472k chars) — this was a legitimate fix but **did not** solve the core issue.
4. Sampled real narrative content across the document at multiple points (5%, 15%, 30%, 50%, 70%, 90%) — confirmed genuine 10-K prose (business description, risk factors, financial notes) exists throughout the document normally.
5. Searched broadly for any "Item"/"PART" text anywhere in the document — found only 27 matches total, **all from the same front-of-document index**.

**Conclusion:** Intel's actual in-body section headings do not contain the literal text "Item N." as a text token — unlike NVDA/AMD/QCOM/AVGO, the visual headings are likely rendered through styling/formatting alone (bold, font size) rather than matchable inline text. No regex over visible text can recover these headers.

**Decision:** rather than reverse-engineer Intel's specific rendering further, added a **fallback strategy**: if Item-based splitting finds fewer than 12 sections for a filing, fall back to generic fixed-size chunking (2,000 chars, 200-char overlap) instead of section-aware chunking.

**Result:** all 4 INTC filings now correctly trigger the fallback and produce 253–267 usable chunks, instead of silently returning broken/incomplete data.

| Filer | Method | Sections/Chunks |
|---|---|---|
| NVDA | section-aware | 19–21 |
| AMD | section-aware | 17–19 |
| QCOM | section-aware | 18–20 |
| AVGO | section-aware | 20–21 |
| INTC | fallback (fixed-size) | 253–267 |

---

## Takeaways so far
- SEC filings are not as standardized as their "Item N." structure suggests — different filers (and likely different filing software vendors) render section headers differently enough to break a naive regex.
- Debugging real messy data required going several layers deep: raw text inspection → hypothesis → targeted verification → correct root cause. The XBRL-hidden-block hypothesis was reasonable and worth fixing, but wasn't the actual cause of INTC's issue — worth noting that not every plausible fix is the *correct* fix.
- A graceful fallback (rather than forcing every filer through the same extraction path) is more robust than continuing to special-case each filer's format indefinitely — this mirrors how production systems handle heterogeneous real-world data.
