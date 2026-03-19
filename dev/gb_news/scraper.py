import hashlib, json, time, random
from pathlib import Path
from typing import List, Tuple, Optional

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
BASE_SLEEP = 1.0          # sleep after EVERY URL (not just successes)
JITTER = 0.6              # add 0..0.6 seconds
CDX_TIMEOUT = (10, 45)    # (connect, read) seconds
SNAP_TIMEOUT = (10, 90)

# -------------------------
# Requests Session (keep-alive + retry)
# -------------------------
session = requests.Session()
session.headers.update(UA)

# HTTP-layer retries for transient failures
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

# Tenacity retries for connection/timeouts at the requests layer
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
    # Longer read timeout for CDX
    j = get(cdx_url, params=params, timeout=CDX_TIMEOUT).json()
    if len(j) <= 1:
        return []
    header = j[0]
    return [dict(zip(header, row)) for row in j[1:]]

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
    stats = {"no_captures": 0, "no_text": 0, "success": 0, "errors": 0}

    with out_path.open("a", encoding="utf-8") as f:
        for i, url in enumerate(urls, 1):
            try:
                print(f"\n[{i}/{len(urls)}] {url}")
                captures = wayback_cdx(url)
                print(f"  📸 Captures: {len(captures)}")

                if not captures:
                    stats["no_captures"] += 1
                    continue

                cap = captures[0]
                ts = cap["timestamp"]
                print(f"  ⏰ Using snapshot: {ts}")

                snap_url, html = fetch_snapshot(url, ts)
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
                    "wayback_url": snap_url,
                    "extraction": extracted,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["success"] += 1
                print("  ✅ Success")

            except Exception as e:
                stats["errors"] += 1
                print(f"  ❌ ERR: {type(e).__name__}: {e}")

            # ✅ always sleep (prevents CDX hammering when many have 0 captures)
            polite_sleep()

    print(f"\n{'='*50}")
    print(f"Results: {stats['success']} success, {stats['no_captures']} no captures, {stats['no_text']} no text, {stats['errors']} errors")

def run_many_sitemaps(start: int, end: int, per_sitemap_limit: Optional[int] = None, total_limit: Optional[int] = None):
    """
    Iterate numbered GB News sitemaps and scrape only /politics/ URLs.
    More robust against timeouts:
      - requests.Session keep-alive
      - HTTP retries + tenacity retries for timeouts/connection errors
      - sleep after EVERY URL attempt
    """
    seen = set()
    stats = {"success": 0, "no_captures": 0, "no_text": 0, "errors": 0, "politics_urls": 0}

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

                    cap = captures[0]
                    ts = cap["timestamp"]
                    print(f"  ⏰ Using snapshot: {ts}")

                    snap_url, html = fetch_snapshot(url, ts)
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
                        "wayback_url": snap_url,
                        "extraction": extracted,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["success"] += 1
                    print("  ✅ Success")

                    if total_limit and stats["success"] >= total_limit:
                        print("\n🛑 Total success limit reached.")
                        print(stats)
                        return

                except Exception as e:
                    stats["errors"] += 1
                    print(f"  ❌ ERR: {type(e).__name__}: {e}")

                # ✅ always sleep, even on errors / no captures
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
    # Example: scan first 50 sitemaps, process up to 5 politics URLs per sitemap,
    # stop after 100 successfully saved articles.
    run_many_sitemaps(start=1, end=2000, per_sitemap_limit=5, total_limit=1000)