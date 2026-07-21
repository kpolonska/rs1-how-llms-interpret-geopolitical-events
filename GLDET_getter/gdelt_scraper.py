import re
import sys
import time
from datetime import datetime, timedelta
import trafilatura
import pandas as pd
import requests
from gdeltdoc import GdeltDoc, Filters
from gdeltdoc.errors import RateLimitError


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def fetch_text(url, timeout=5):
    try:
        resp = requests.get(url, timeout=timeout, headers=HEADERS, allow_redirects=True)
        if resp.status_code >= 400:
            return None
        return trafilatura.extract(resp.text)
    except Exception:
        return None


MIN_CHARS = 500
KEYWORD_RE = re.compile(r"\b(?:ceasefire|cease.fire|truce|armistice)\b", re.IGNORECASE)


def _norm_title(t: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for title comparison."""
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t)


def clean(df: pd.DataFrame):
    report = []
    n0 = len(df)

    # 1. Too short — likely a snippet, paywall stub, or nav boilerplate
    df = df[df["full_text"].str.len() >= MIN_CHARS].copy()
    report.append(("Too short (<500 chars)", n0 - len(df)))
    n0 = len(df)

    # 2. No keyword in extracted text — trafilatura grabbed the wrong content
    df = df[df["full_text"].str.contains(KEYWORD_RE)].copy()
    report.append(("No ceasefire keyword in text", n0 - len(df)))
    n0 = len(df)

    # 3. Duplicate titles (same story, different syndication domains) — keep longest text
    df["_title_norm"] = df["title"].apply(_norm_title)
    df = (
        df.sort_values("full_text", key=lambda s: s.str.len(), ascending=False)
          .drop_duplicates(subset="_title_norm", keep="first")
          .drop(columns="_title_norm")
          .sort_index()
    )
    report.append(("Duplicate titles (kept longest)", n0 - len(df)))

    return df.reset_index(drop=True), report


def gdelt_query_with_retry(gd, f, max_attempts=6):
    for attempt in range(max_attempts):
        if attempt > 0:
            wait = 10 * attempt
            print(f"Waiting {wait}s before retry {attempt}/{max_attempts - 1}...")
            time.sleep(wait)
        try:
            result = gd.article_search(f)
            return result
        except RateLimitError:
            print(f"Rate limited (attempt {attempt + 1}).")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Network error ({e.__class__.__name__}) on attempt {attempt + 1}.")
    print("GDELT API unavailable after all retries.")
    sys.exit(1)


def main():
    print("Querying GDELT DOC 2.0 API...")

    end = datetime.utcnow()
    start = end - timedelta(days=90)

    gd = GdeltDoc()
    f = Filters(
        keyword="ceasefire",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        num_records=250,
        language="English",
    )
    print(f"Query string: {f.query_string}")

    # Initial 6-second wait to respect the API's 5s rate-limit window
    time.sleep(6)
    articles = gdelt_query_with_retry(gd, f)
    print(f"GDELT returned {len(articles)} articles.\n")

    if articles.empty:
        print("No articles returned. Exiting.")
        sys.exit(0)

    keep_cols = ["title", "url", "seendate", "domain", "sourcecountry"]
    missing = [c for c in keep_cols if c not in articles.columns]
    if missing:
        print(f"Warning: columns missing from GDELT response: {missing}")
    meta = articles[[c for c in keep_cols if c in articles.columns]].copy()

    total = len(meta)
    full_texts = []
    failed = 0

    for i, row in enumerate(meta.itertuples(), start=1):
        text = fetch_text(row.url)
        full_texts.append(text)
        if text is None:
            failed += 1
        if i % 10 == 0 or i == total:
            print(f"Scraped {i}/{total}... (failed so far: {failed})")

    meta["full_text"] = full_texts

    after_scrape = len(meta)
    meta = meta[meta["full_text"].notna() & (meta["full_text"].str.strip() != "")]
    print(f"\nDropped {after_scrape - len(meta)} rows with empty/None full_text.")

    meta, cleaning_report = clean(meta)

    # Flatten paragraph breaks so every CSV row is exactly one line
    meta["full_text"] = meta["full_text"].str.replace(r"\n+", " ", regex=True).str.strip()

    meta.to_csv("gdelt_articles.csv", index=False)

    # JSONL: one compact JSON object per line
    with open("gdelt_articles.jsonl", "w", encoding="utf-8") as f:
        for rec in meta.to_dict(orient="records"):
            f.write(__import__("json").dumps(rec, ensure_ascii=False) + "\n")

    print("Saved gdelt_articles.csv and gdelt_articles.jsonl.\n")

    print("=== Summary ===")
    print(f"Total articles found       : {total}")
    print(f"Successfully scraped       : {after_scrape}")
    print(f"Failed / skipped fetching  : {failed}")
    for label, n in cleaning_report:
        print(f"  {label:<30}: -{n}")
    print(f"Final clean articles       : {len(meta)}")
    print()
    print("First 3 titles + domains:")
    for _, row in meta.head(3).iterrows():
        title = row.get("title", "N/A")
        domain = row.get("domain", "N/A")
        print(f"  [{domain}] {title}")


if __name__ == "__main__":
    main()
