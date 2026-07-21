"""
Fetches 50 Russian-language political/geopolitical news articles from RSS feeds.
Sources: Meduza, BBC Russian, Zona.media, UNIAN Russian, NV.ua
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

RSS_FEEDS = [
    ("meduza.io",   "https://meduza.io/rss/all"),
    ("bbc-russian", "https://feeds.bbci.co.uk/russian/rss.xml"),
    ("zona.media",  "https://zona.media/rss"),
    ("unian.net",   "https://www.unian.net/rss"),
    ("nv.ua",       "https://nv.ua/rss/all.xml"),
]

# ── Geopolitical / political filter ─────────────────────────────────────────

EXCLUDE = re.compile(
    r"(?:"
    # sports
    r"футбол|матч|чемпионат|сборн\w+\s+(?:по|России)|гол\b|трансфер|клуб|стадион"
    r"|"
    # entertainment / celebrity
    r"(?:звезда|актер|певец|певица|артист)\s+(?:сериала|фильма|шоу)|сериал\w*\s+(?:расска|призна|заявил)"
    r"|"
    # pure science/tech without political context
    r"квантов\w+\s+компьютер|магнитн\w+\s+волн|размером с монету"
    r"|"
    # agricultural subsidies / farming without political angle
    r"(?:украинские|российские)\s+хозяйства\s+получили\s+выплаты\s+(?:на\s+содержание|для)"
    r"|"
    # gossip / illness / personal life
    r"тяжел\w+\s+болезн|рассказала о (?:болезни|романе|личной)"
    r")",
    re.IGNORECASE,
)

INCLUDE = re.compile(
    r"(?:"
    # war / military operations
    r"война|военн\w+\s+(?:операци|удар|наступ|действи)|обстрел|ракетн|удар\s+по|атак\w+\s+(?:Киев|Харьков|Одесс|Николаев|Херсон|Запорожь|Донецк|Луганск|Мариуп|Белгород|Курск)"
    r"|"
    # Ukraine conflict
    r"Украин\w+\s+(?:фронт|армия|ВСУ|войск|наступ|оборон)|ВСУ\b|Зеленск"
    r"|"
    # Russian military / politics
    r"Путин|Кремл|Минобороны\s+РФ|российск\w+\s+(?:войск|армия|наступ|удар|ракет)"
    r"|"
    # political institutions / governance
    r"парламент|Государственная\s+Дума|Совет\s+Федерации|правительств\w+\s+(?:России|Украины|Беларуси|решил|принял)|президент\w*\s+(?:подписал|заявил|встретился|приказал)"
    r"|"
    # geopolitical / international
    r"НАТО|санкци|переговор\w+|дипломат|союзник|Евросоюз|\bЕС\b\s+(?:ввел|принял|решил)|G7|G20"
    r"|"
    # occupation / territorial
    r"оккупи|аннексир|временно\s+оккупирован|Крым\w*\s+(?:получил|власт|админист)|оккупант"
    r"|"
    # political repression / espionage
    r"госизмен|иностранн\w+\s+агент|шпионаж|политическ\w+\s+(?:заключен|преследован|арест)|ФСБ\s+(?:задержал|арест|обвинил)"
    r"|"
    # war crimes / international justice
    r"военн\w+\s+преступлени|МУС|трибунал|(?:задержан|арестован|обвиняется)\s+.{0,30}(?:войн|Украин|Россия)"
    r"|"
    # Belarus / regional geopolitics
    r"Лукашенк|Беларус\w+\s+(?:армия|политик|санкци|выборы|протест)"
    r"|"
    # Iran/Middle East/other conflicts referenced in Russian press
    r"Хаменеи|Иран\w*\s+(?:Украина|санкци|ядерн|переговор)|Израиль\w*\s+(?:войн|удар|операц)"
    r"|"
    # госдолг / debt when geopolitical
    r"Госдолг\s+Украины"
    r")",
    re.IGNORECASE,
)


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
                "title":         title,
                "url":           link,
                "seendate":      date,
                "domain":        urlparse(link).netloc,
                "sourcecountry": "RU",
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


def is_geopolitical(row) -> bool:
    haystack = row["title"] + " " + str(row["full_text"])[:400]
    if EXCLUDE.search(haystack):
        return False
    return bool(INCLUDE.search(haystack))


def clean(df: pd.DataFrame):
    report = []
    n0 = len(df)

    df = df[df["full_text"].str.len() >= MIN_CHARS].copy()
    report.append(("Too short (<500 chars)", n0 - len(df))); n0 = len(df)

    print(f"  Detecting language on {len(df)} candidates...")
    df["lang"] = df["full_text"].apply(detect_lang)
    non_ru = (df["lang"] != "ru").sum()
    df = df[df["lang"] == "ru"].drop(columns="lang").copy()
    report.append(("Not Russian (langdetect)", non_ru)); n0 = len(df)

    df["keep"] = df.apply(is_geopolitical, axis=1)
    not_geo = (~df["keep"]).sum()
    df = df[df["keep"]].drop(columns="keep").copy()
    report.append(("Not political/geopolitical", not_geo)); n0 = len(df)

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
    print("Collecting URLs from Russian-language RSS feeds...\n")
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
    meta["full_text"] = meta["full_text"].str.replace(r"\n+", " ", regex=True).str.strip()

    if len(meta) > TARGET_CLEAN:
        meta = meta.head(TARGET_CLEAN).reset_index(drop=True)

    meta.to_csv("gdelt_russian_articles.csv", index=False)
    with open("gdelt_russian_articles.jsonl", "w", encoding="utf-8") as f:
        for rec in meta.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nSaved gdelt_russian_articles.csv and gdelt_russian_articles.jsonl.")
    print("\n=== Summary ===")
    print(f"URLs from feeds           : {total_found}")
    print(f"Failed fetching           : {failed}")
    for label, n in report:
        print(f"  {label:<35}: -{n}")
    print(f"Final clean articles      : {len(meta)}")
    if len(meta) < TARGET_CLEAN:
        print(f"  Note: only {len(meta)} found. Consider adding more RSS feeds.")
    print()
    print("First 5 titles + domains:")
    for _, row in meta.head(5).iterrows():
        sys.stdout.buffer.write(
            f"  [{row.get('domain','?')}] {row.get('title','?')}\n".encode("utf-8")
        )


if __name__ == "__main__":
    main()
