import hashlib, json, time, random
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lxml import etree
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from trafilatura import extract as traf_extract
from bs4 import BeautifulSoup

# -------------------------
# Paths / config
# -------------------------
DATA = Path("gbnews_corpus/data")
RAW = DATA / "raw_html"
OUT = DATA / "extracted_jsonl"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "AcademicResearchBot/1.0 (contact: your-email)"}
POLITICS_PREFIX = "https://www.gbnews.com/politics/"

# Politeness / stability
BASE_SLEEP = 1.0          # sleep after EVERY URL attempt
JITTER = 0.6              # add 0..0.6 seconds
CDX_TIMEOUT = (10, 45)    # (connect, read) seconds
SNAP_TIMEOUT = (10, 90)

# Date filtering (by PUBLISHED date if available; fallback to Wayback date)
TARGET_FROM_YEAR = 2014
TARGET_TO_YEAR = 2026

# Capture selection: "earliest" or "latest"
CAPTURE_POLICY = "earliest"

# -------------------------
# Requests Session (keep-alive + retry)
# -------------------------
session = requests.Session()
session.headers.update(UA)

http_retries = Retry(
    total=5,
    connect=5,
    read=5,
    status=5,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=http_retries, pool_connections=20, pool_maxsize=20)
session.mount("https://", adapter)
session.mount("http://", adapter)

def polite_sleep(base: float = BASE_SLEEP) -> None:
    time.sleep(base + random.random() * JITTER)

RETRY_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)

@retry(
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    wait=wait_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
)
def get(url: str, params=None, timeout: Tuple[int, int] = (10, 60)) -> requests.Response:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r

# -------------------------
# Helpers
# -------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def is_politics_url(url: str) -> bool:
    return url.startswith(POLITICS_PREFIX)

def wayback_ts_to_iso(ts: str) -> str:
    # Wayback ts: YYYYMMDDhhmmss (UTC)
    dt = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return dt.isoformat()

def normalize_date_to_iso(date_str: str) -> Optional[str]:
    """
    Best-effort normalization to ISO8601.
    Returns None if parsing fails.
    """
    if not date_str:
        return None
    s = date_str.strip()

    # ISO-like strings
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        pass

    # Simple common formats (date-only)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            continue

    return None

def year_from_iso(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).year
    except Exception:
        return None

def _first_str(x):
    if isinstance(x, list) and x:
        return _first_str(x[0])
    return x if isinstance(x, str) else None

def extract_published_date(html: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract published date from HTML (best-effort).
    Returns (published_raw, source).
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) JSON-LD (often most reliable)
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.get_text(strip=True))
        except Exception:
            continue

        candidates = []
        if isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                candidates.extend([x for x in data["@graph"] if isinstance(x, dict)])
            candidates.append(data)
        elif isinstance(data, list):
            candidates.extend([x for x in data if isinstance(x, dict)])

        for obj in candidates:
            dt = _first_str(obj.get("datePublished")) or _first_str(obj.get("dateCreated"))
            if dt:
                return dt, "jsonld:datePublished/dateCreated"

    # 2) OpenGraph
    og = soup.find("meta", attrs={"property": "article:published_time"})
    if og and og.get("content"):
        return og["content"].strip(), "meta:article:published_time"

    # 3) Common meta name variants
    for name in ["pubdate", "publish-date", "publication_date", "date", "Date", "DC.date.issued"]:
        m = soup.find("meta", attrs={"name": name})
        if m and m.get("content"):
            return m["content"].strip(), f"meta:name:{name}"

    # 4) <time datetime="...">
    t = soup.find("time")
    if t:
        if t.get("datetime"):
            return t["datetime"].strip(), "time:datetime"
        txt = t.get_text(" ", strip=True)
        if txt:
            return txt, "time:text"

    return None, None

def parse_sitemap(sitemap_url: str) -> list[str]:
    print(f"🔎 Fetching sitemap: {sitemap_url}")
    xml = get(sitemap_url, timeout=CDX_TIMEOUT).content
    print(f"  📦 Downloaded {len(xml)} bytes")

    root = etree.fromstring(xml)
    ns = root.nsmap.get(None, "")
    nsmap = {"ns": ns} if ns else None

    # sitemap index
    if root.tag.endswith("sitemapindex"):
        print("  📂 This is a sitemap index")
        locs = (
            root.xpath("//ns:sitemap/ns:loc/text()", namespaces=nsmap)
            if nsmap else
            root.xpath("//sitemap/loc/text()")
        )
        print(f"  📑 Found {len(locs)} child sitemaps")

        urls = []
        for loc in locs:
            print(f"  ↪ Expanding child sitemap: {loc}")
            urls.extend(parse_sitemap(loc))
        return urls

    # urlset
    print("  📰 This is a URL set")
    locs = (
        root.xpath("//ns:url/ns:loc/text()", namespaces=nsmap)
        if nsmap else
        root.xpath("//url/loc/text()")
    )
    print(f"  🔗 Found {len(locs)} URLs in this sitemap")
    return [u.strip() for u in locs]

def wayback_cdx(url: str) -> list[dict]:
    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": url,
        "output": "json",
        "filter": "statuscode:200",
        "collapse": "digest",
        "fl": "timestamp,original,statuscode,mimetype,digest,length",
    }
    j = get(cdx_url, params=params, timeout=CDX_TIMEOUT).json()
    if len(j) <= 1:
        return []
    header = j[0]
    return [dict(zip(header, row)) for row in j[1:]]

def choose_capture(captures: list[dict], policy: str = CAPTURE_POLICY) -> Optional[dict]:
    if not captures:
        return None
    return captures[0] if policy == "earliest" else captures[-1]

def fetch_snapshot(original_url: str, timestamp: str) -> tuple[str, str]:
    snap_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
    html = get(snap_url, timeout=SNAP_TIMEOUT).text
    return snap_url, html

def extract_text(html: str) -> dict:
    text = traf_extract(html, include_comments=False, include_tables=False)
    if text and len(text) > 200:
        return {"text": text, "method": "trafilatura"}

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else None)
    body = soup.get_text("\n", strip=True)
    return {"text": body if len(body) > 200 else None, "title": title, "method": "bs4_fallback"}

def date_in_target_window(published_iso: Optional[str], wayback_ts: str) -> bool:
    """
    Prefer filtering by published date if parseable; else fall back to Wayback year.
    """
    y = year_from_iso(published_iso)
    if y is None:
        y = int(wayback_ts[:4])
    return TARGET_FROM_YEAR <= y <= TARGET_TO_YEAR

# -------------------------
# Main runners
# -------------------------
def run(sitemap_url: str, limit: Optional[int] = None):
    print(f"🚀 Starting scrape: {sitemap_url}")

    urls = parse_sitemap(sitemap_url)
    print(f"📋 Total URLs discovered: {len(urls)}")

    urls = [u for u in urls if is_politics_url(u)]
    print(f"🏛️ Politics URLs after filter: {len(urls)}")

    if limit:
        urls = urls[:limit]
        print(f"🔢 Limiting to first {len(urls)} URLs")

    out_path = OUT / "gbnews_politics.jsonl"
    stats = {
        "no_captures": 0, "no_text": 0, "success": 0, "errors": 0,
        "skipped_outside_years": 0, "missing_published_date": 0
    }

    with out_path.open("a", encoding="utf-8") as f:
        for i, url in enumerate(urls, 1):
            try:
                print(f"\n[{i}/{len(urls)}] {url}")
                captures = wayback_cdx(url)
                print(f"  📸 Captures: {len(captures)}")

                if not captures:
                    stats["no_captures"] += 1
                    continue

                cap = choose_capture(captures)
                ts = cap["timestamp"]
                print(f"  ⏰ Using snapshot: {ts} ({CAPTURE_POLICY})")

                snap_url, html = fetch_snapshot(url, ts)

                published_raw, published_source = extract_published_date(html)
                published_iso = normalize_date_to_iso(published_raw) if published_raw else None
                if not published_raw:
                    stats["missing_published_date"] += 1

                if not date_in_target_window(published_iso, ts):
                    stats["skipped_outside_years"] += 1
                    print(f"  ⏭️ Skipping (outside {TARGET_FROM_YEAR}-{TARGET_TO_YEAR})")
                    continue

                key = sha1(f"{url}|{ts}")
                (RAW / f"{key}.html").write_text(html, encoding="utf-8")

                extracted = extract_text(html)
                text_len = len(extracted.get("text", "")) if extracted.get("text") else 0
                print(f"  📝 Extracted text: {text_len} chars ({extracted.get('method', 'none')})")

                if not extracted.get("text"):
                    stats["no_text"] += 1
                    continue

                rec = {
                    "source": "gbnews",
                    "section_filter": "politics",
                    "original_url": url,
                    "wayback_timestamp": ts,
                    "wayback_datetime_utc": wayback_ts_to_iso(ts),
                    "published_raw": published_raw,
                    "published_datetime": published_iso,
                    "published_source": published_source,
                    "wayback_url": snap_url,
                    "extraction": extracted,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["success"] += 1
                print(f"  ✅ Success | published={published_iso or published_raw or 'unknown'}")

            except Exception as e:
                stats["errors"] += 1
                print(f"  ❌ ERR: {type(e).__name__}: {e}")

            polite_sleep()

    print(f"\n{'='*50}")
    print(
        f"Results: {stats['success']} success, {stats['no_captures']} no captures, "
        f"{stats['no_text']} no text, {stats['skipped_outside_years']} skipped(outside years), "
        f"{stats['errors']} errors"
    )

def run_many_sitemaps(start: int, end: int, per_sitemap_limit: Optional[int] = None, total_limit: Optional[int] = None):
    """
    Iterate numbered GB News sitemaps and scrape only /politics/ URLs.
    Adds:
      - robust networking
      - always-sleep throttling
      - published date extraction + year filtering (2014-2024 by default)
      - optional stop after N successes
    """
    seen = set()
    stats = {
        "success": 0, "no_captures": 0, "no_text": 0, "errors": 0,
        "politics_urls": 0, "skipped_outside_years": 0, "missing_published_date": 0
    }

    out_path = OUT / "gbnews_politics.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        for n in range(start, end + 1):
            sitemap_url = f"https://www.gbnews.com/feeds/sitemaps/sitemap_{n}.xml"
            print(f"\n====================")
            print(f"🧭 Sitemap {n}: {sitemap_url}")

            try:
                urls = parse_sitemap(sitemap_url)
            except Exception as e:
                stats["errors"] += 1
                print(f"  ❌ Failed to fetch/parse sitemap_{n}: {type(e).__name__}: {e}")
                polite_sleep(2.0)
                continue

            politics_urls = [u for u in urls if is_politics_url(u)]
            politics_urls = [u for u in politics_urls if u not in seen]
            for u in politics_urls:
                seen.add(u)

            stats["politics_urls"] += len(politics_urls)
            print(f"  🏛️ Politics URLs in this sitemap (new): {len(politics_urls)}")

            if per_sitemap_limit:
                politics_urls = politics_urls[:per_sitemap_limit]

            for i, url in enumerate(politics_urls, 1):
                try:
                    print(f"\n[{i}/{len(politics_urls)}] {url}")
                    captures = wayback_cdx(url)
                    print(f"  📸 Captures: {len(captures)}")

                    if not captures:
                        stats["no_captures"] += 1
                        continue

                    cap = choose_capture(captures)
                    ts = cap["timestamp"]
                    print(f"  ⏰ Using snapshot: {ts} ({CAPTURE_POLICY})")

                    snap_url, html = fetch_snapshot(url, ts)

                    published_raw, published_source = extract_published_date(html)
                    published_iso = normalize_date_to_iso(published_raw) if published_raw else None
                    if not published_raw:
                        stats["missing_published_date"] += 1

                    if not date_in_target_window(published_iso, ts):
                        stats["skipped_outside_years"] += 1
                        print(f"  ⏭️ Skipping (outside {TARGET_FROM_YEAR}-{TARGET_TO_YEAR})")
                        continue

                    key = sha1(f"{url}|{ts}")
                    (RAW / f"{key}.html").write_text(html, encoding="utf-8")

                    extracted = extract_text(html)
                    text_len = len(extracted.get("text", "")) if extracted.get("text") else 0
                    print(f"  📝 Extracted text: {text_len} chars ({extracted.get('method', 'none')})")

                    if not extracted.get("text"):
                        stats["no_text"] += 1
                        continue

                    rec = {
                        "source": "gbnews",
                        "section_filter": "politics",
                        "original_url": url,
                        "wayback_timestamp": ts,
                        "wayback_datetime_utc": wayback_ts_to_iso(ts),
                        "published_raw": published_raw,
                        "published_datetime": published_iso,
                        "published_source": published_source,
                        "wayback_url": snap_url,
                        "extraction": extracted,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["success"] += 1
                    print(f"  ✅ Success | published={published_iso or published_raw or 'unknown'}")

                    if total_limit and stats["success"] >= total_limit:
                        print("\n🛑 Total success limit reached.")
                        print(stats)
                        return

                except Exception as e:
                    stats["errors"] += 1
                    print(f"  ❌ ERR: {type(e).__name__}: {e}")

                polite_sleep()

            print(f"\n📊 Running totals: {stats}")

def list_child_sitemaps(sitemap_index_url: str) -> List[str]:
    xml = get(sitemap_index_url, timeout=CDX_TIMEOUT).content
    root = etree.fromstring(xml)
    ns = root.nsmap.get(None, "")
    nsmap = {"ns": ns} if ns else None

    if not root.tag.endswith("sitemapindex"):
        raise ValueError(f"Not a sitemapindex: {sitemap_index_url}")

    locs = (
        root.xpath("//ns:sitemap/ns:loc/text()", namespaces=nsmap)
        if nsmap else
        root.xpath("//sitemap/loc/text()")
    )
    return [l.strip() for l in locs]

def count_numbered_sitemaps() -> int:
    idx = list_child_sitemaps("https://www.gbnews.com/sitemap.xml")
    numbered = [u for u in idx if "/feeds/sitemaps/sitemap_" in u and u.endswith(".xml")]
    print(f"✅ Found {len(numbered)} numbered sitemap files referenced by sitemap.xml")
    return len(numbered)

if __name__ == "__main__":
    # Note: GB News (the outlet) is newer than 2014, so you may see many "skipped_outside_years"
    # if Wayback captures are mostly from later years. Adjust TARGET_FROM_YEAR/TARGET_TO_YEAR as needed.
    run_many_sitemaps(start=1, end=2000, per_sitemap_limit=5, total_limit=1000)