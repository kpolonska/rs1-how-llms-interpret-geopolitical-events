"""
Fetches 50 Ukrainian-language geopolitical/war news articles directly from
RSS feeds of major Ukrainian outlets (bypasses GDELT, which has limited .ua
coverage and rate-limits aggressively).

Sources confirmed reachable:
  - suspilne.media  (Suspilne public broadcaster, Ukrainian)
  - 24tv.ua         (24 Kanal news, Ukrainian)
"""
import re
import sys
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
import pandas as pd
import trafilatura
from langdetect import detect, LangDetectException


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

MIN_CHARS = 500
TARGET_CLEAN = 50

# Ukrainian war/conflict terms — article body must contain at least one
KEYWORD_RE = re.compile(
    r"(?:війн|бойов|обстріл|наступ|окупант|фронт|збройн|ЗСУ|армі|загибл|ракет|удар|втрат|Росі|агресі|полон|поранен)",
    re.IGNORECASE,
)

RSS_FEEDS = [
    ("suspilne.media",  "https://suspilne.media/rss/all.rss"),
    ("suspilne-pol",    "https://suspilne.media/rss/politics.rss"),
    ("24tv.ua",         "https://24tv.ua/rss/all.xml"),
]


# ── RSS parsing ──────────────────────────────────────────────────────────────

def parse_rss(content: bytes) -> list[dict]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link")  or "").strip()
        date  = (item.findtext("pubDate") or "").strip()
        if link.startswith("http"):
            items.append({
                "title":       title,
                "url":         link,
                "seendate":    date,
                "domain":      urlparse(link).netloc,
                "sourcecountry": "Ukraine",
            })
    return items


def fetch_feed(name: str, url: str) -> list[dict]:
    try:
        r = requests.get(url, timeout=10, headers=HEADERS)
        if r.status_code != 200:
            print(f"  [{name}] HTTP {r.status_code}")
            return []
        items = parse_rss(r.content)
        print(f"  [{name}] {len(items)} URLs")
        return items
    except Exception as e:
        print(f"  [{name}] {type(e).__name__}")
        return []


# ── Article scraping ─────────────────────────────────────────────────────────

def fetch_text(url: str, timeout: int = 7) -> str | None:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS, allow_redirects=True)
        if r.status_code >= 400:
            return None
        return trafilatura.extract(r.text)
    except Exception:
        return None


# ── Cleaning ─────────────────────────────────────────────────────────────────

def detect_lang(text: str) -> str:
    try:
        return detect(text[:1000])
    except LangDetectException:
        return "unknown"


def _norm_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t)


def clean(df: pd.DataFrame):
    report = []
    n0 = len(df)

    df = df[df["full_text"].str.len() >= MIN_CHARS].copy()
    report.append(("Too short (<500 chars)", n0 - len(df))); n0 = len(df)

    df = df[df["full_text"].str.contains(KEYWORD_RE)].copy()
    report.append(("No conflict keywords", n0 - len(df))); n0 = len(df)

    print(f"  Detecting language on {len(df)} candidates...")
    df["lang"] = df["full_text"].apply(detect_lang)
    non_uk = (df["lang"] != "uk").sum()
    df = df[df["lang"] == "uk"].drop(columns="lang").copy()
    report.append(("Not Ukrainian (langdetect)", non_uk)); n0 = len(df)

    df["_norm"] = df["title"].apply(_norm_title)
    df = (
        df.sort_values("full_text", key=lambda s: s.str.len(), ascending=False)
          .drop_duplicates(subset="_norm", keep="first")
          .drop(columns="_norm")
          .sort_index()
          .reset_index(drop=True)
    )
    report.append(("Duplicate titles (kept longest)", n0 - len(df)))
    return df, report


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Collecting URLs from Ukrainian RSS feeds...")
    seen_urls: set[str] = set()
    all_items: list[dict] = []

    for name, url in RSS_FEEDS:
        for item in fetch_feed(name, url):
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)

    total_found = len(all_items)
    print(f"\nTotal unique URLs: {total_found}")
    if not total_found:
        print("No URLs collected. Exiting.")
        sys.exit(0)

    meta = pd.DataFrame(all_items)

    print("\nScraping articles...")
    full_texts, failed = [], 0
    for i, row in enumerate(meta.itertuples(), start=1):
        text = fetch_text(row.url)
        full_texts.append(text)
        if text is None:
            failed += 1
        if i % 10 == 0 or i == total_found:
            print(f"Scraped {i}/{total_found}... (failed: {failed})")

    meta["full_text"] = full_texts
    before = len(meta)
    meta = meta[meta["full_text"].notna() & (meta["full_text"].str.strip() != "")]
    print(f"\nDropped {before - len(meta)} rows with empty full_text.")

    print("\nCleaning...")
    meta, report = clean(meta)

    # One article per line
    meta["full_text"] = meta["full_text"].str.replace(r"\n+", " ", regex=True).str.strip()

    if len(meta) > TARGET_CLEAN:
        meta = meta.head(TARGET_CLEAN).reset_index(drop=True)

    meta.to_csv("gdelt_ukraine_articles.csv", index=False)
    with open("gdelt_ukraine_articles.jsonl", "w", encoding="utf-8") as f:
        for rec in meta.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nSaved gdelt_ukraine_articles.csv and gdelt_ukraine_articles.jsonl.")
    print("\n=== Summary ===")
    print(f"URLs from feeds           : {total_found}")
    print(f"Failed fetching           : {failed}")
    for label, n in report:
        print(f"  {label:<35}: -{n}")
    print(f"Final clean articles      : {len(meta)}")
    print()
    print("First 5 titles + domains:")
    for _, row in meta.head(5).iterrows():
        sys.stdout.buffer.write(
            f"  [{row.get('domain','?')}] {row.get('title','?')}\n".encode("utf-8")
        )


if __name__ == "__main__":
    main()
