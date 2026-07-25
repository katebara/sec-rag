"""
Download 10-K filings for a set of companies from SEC EDGAR.
"""

import os
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

load_dotenv()

COMPANY_NAME = os.getenv("katebara", "Anonymous")
EMAIL = os.getenv("katebaranovaa@gmail.com", "anonymous@example.com")

TICKERS = ["NVDA", "AMD", "INTC", "QCOM", "AVGO"]
DOWNLOAD_FOLDER = "data/raw"

def download_filings(tickers, after="2022-01-01", before="2026-01-01"):
    dl = Downloader(COMPANY_NAME, EMAIL, DOWNLOAD_FOLDER)
    for ticker in tickers:
        print(f"Downloading 10-Ks for {ticker}...")
        dl.get("10-K", ticker, after=after, before=before)
        print(f"Done with {ticker}")

if __name__ == "__main__":
    download_filings(TICKERS)
