#!/usr/bin/env python3
"""
MP Gazetteer Builder
====================
Collects names of British MPs (House of Commons, House=1) who served
between 1990 and 2024 from the UK Parliament Members API, normalizes
names, resolves titled/peerage names back to real names using
BeautifulSoup4 to parse biography pages, and writes the final
gazetteer to mp_gazetteer.csv.

Usage:
    pip install requests beautifulsoup4
    python mp_gazetteer.py

Output:  mp_gazetteer.csv
"""

import csv
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ─── Configuration ───────────────────────────────────────────────────
API_BASE = "https://members-api.parliament.uk/api"
SEARCH_URL = f"{API_BASE}/Members/Search"
BIOGRAPHY_URL = "https://members.parliament.uk/member"  # /{id}/career
PAGE_SIZE = 20  # API max per request
RATE_LIMIT_DELAY = 0.3  # seconds between API calls (be polite)
BIO_RATE_LIMIT = 0.5  # seconds between biography page fetches
DATE_FLOOR = datetime(1990, 1, 1)
DATE_CEIL = datetime(2024, 12, 31)
OUTPUT_FILE = "mp_gazetteer.csv"

# Titles / honorifics to strip during normalization
TITLES_TO_STRIP = [
    r"\bRt\s+Hon\.?\b", r"\bThe\b", r"\bSir\b", r"\bDame\b",
    r"\bLord\b", r"\bLady\b", r"\bBaroness\b", r"\bBaron\b",
    r"\bViscount(?:ess)?\b", r"\bEarl\b", r"\bCount(?:ess)?\b",
    r"\bMarquess\b", r"\bDuke\b", r"\bDuchess\b",
    r"\bMrs?\b", r"\bMs\b", r"\bDr\b", r"\bProf(?:essor)?\b",
    r"\bRev(?:erend)?\.?\b",
    # Post-nominal letters
    r"\b(?:MP|KBE|CBE|OBE|MBE|KG|CH|PC|QC|KC|DL|JP|FRS|FRSE)\b",
    r"\bDSO\b", r"\bMC\b", r"\bDFC\b", r"\bGCB\b", r"\bKCB\b",
    r"\bGCMG\b", r"\bKCMG\b", r"\bKCVO\b", r"\bDBE\b", r"\bGBE\b",
]

# Patterns that indicate a peerage/titled name rather than a personal name
PEERAGE_INDICATORS = [
    # "Baroness of Maidenhead", "Lord of X"
    r"^(?:Lord|Lady|Baron(?:ess)?|Viscount(?:ess)?|Earl|Marquess|Duke|Duchess)\s+of\s+",
    # "Baroness May of Maidenhead", "Lord Cameron of Chipping Norton" — Title Name of Place
    r"^(?:Lord|Lady|Baron(?:ess)?|Viscount(?:ess)?|Earl|Marquess|Duke|Duchess)\s+\w+\s+of\s+",
    # "Lord Mandelson", "Baroness Thatcher" — Title + single word (no place)
    r"^(?:Lord|Lady|Baron(?:ess)?)\s+\w+$",
    # nameListAs pattern like "Aberdare, L." or "May of Maidenhead, B."
    r",\s*[LBVEMDC]\.",
]

# ─── Helpers ─────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def safe_get(url, params=None, headers=None, retries=3, delay=1.0):
    """GET with retries and back-off."""
    default_headers = {"Accept": "application/json", "User-Agent": "MP-Gazetteer/1.0"}
    if headers:
        default_headers.update(headers)
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=default_headers, timeout=30)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = delay * (2 ** attempt)
                log(f"Rate-limited (429). Waiting {wait:.0f}s...", "WARN")
                time.sleep(wait)
            else:
                log(f"HTTP {resp.status_code} for {url} (attempt {attempt+1})", "WARN")
                time.sleep(delay)
        except requests.RequestException as e:
            log(f"Request error: {e} (attempt {attempt+1})", "WARN")
            time.sleep(delay)
    return None


# ─── Stage 1: Collect all MPs from the API ───────────────────────────

def _paginate_house(house_num, label):
    """Paginate through the Members Search API for one House."""
    all_items = []
    skip = 0

    params = {"House": house_num, "skip": skip, "take": PAGE_SIZE}
    resp = safe_get(SEARCH_URL, params=params)
    if not resp:
        log(f"Failed to reach the Parliament API for {label}. Check your network.", "ERROR")
        return []

    data = resp.json()
    total = data.get("totalResults", 0)
    log(f"  {label}: API reports {total} total members")

    items = data.get("items", [])
    all_items.extend(items)
    skip += PAGE_SIZE

    while skip < total:
        params = {"House": house_num, "skip": skip, "take": PAGE_SIZE}
        resp = safe_get(SEARCH_URL, params=params)
        if resp:
            page_data = resp.json()
            all_items.extend(page_data.get("items", []))
            if len(all_items) % 200 == 0 or skip + PAGE_SIZE >= total:
                log(f"    ...collected {len(all_items)} / {total} {label} records")
        else:
            log(f"    Skipped batch at offset {skip} due to fetch failure", "WARN")
        skip += PAGE_SIZE
        time.sleep(RATE_LIMIT_DELAY)

    return all_items


def fetch_all_members():
    """
    Fetch members from BOTH houses.

    WHY: The API's House parameter filters by *latest* house membership.
    Former MPs who were elevated to the Lords (e.g. Theresa May → Baroness
    May of Maidenhead, David Cameron → Lord Cameron of Chipping Norton)
    would be missed if we only query House=1.

    Phase A: House=1 (Commons) — gives us all members whose latest
             membership is the Commons.
    Phase B: House=2 (Lords) — we'll check each for prior Commons
             service in filter_by_date_range().
    """
    log("Stage 1: Fetching MP records from Parliament API...")
    log("  Phase A: House of Commons members (House=1)")
    commons = _paginate_house(1, "House of Commons")

    log("  Phase B: House of Lords members (House=2)")
    log("  (to catch former MPs elevated to the peerage)")
    lords = _paginate_house(2, "House of Lords")

    log(f"Stage 1 fetching complete: {len(commons)} Commons + {len(lords)} Lords = {len(commons) + len(lords)} total")
    return commons, lords


def _fetch_all_former_commons_from_mnis():
    """
    Use the older MNIS API (data.parliament.uk) to get a set of member IDs
    for everyone who EVER sat in the House of Commons.

    The MNIS API with criterion 'house*Commons|membership=all' returns
    all current and former Commons members, including those now in the Lords.
    This is far more reliable than scraping 2,788 individual career pages.

    Returns a dict mapping member_id -> {start, end, constituency} for
    their Commons service.
    """
    MNIS_URL = (
        "http://data.parliament.uk/membersdataplatform/services/mnis/"
        "members/query/house*Commons|membership=all/HouseMemberships/"
    )

    log("    Querying MNIS API for all historical Commons members...")
    resp = safe_get(
        MNIS_URL,
        headers={"Accept": "application/json", "User-Agent": "MP-Gazetteer/1.0"},
    )
    if not resp:
        log("    MNIS API unavailable — falling back to Synopsis API method", "WARN")
        return None

    try:
        # The MNIS API often returns a UTF-8 BOM (\xef\xbb\xbf) which
        # breaks resp.json(). Decode with utf-8-sig to strip it.
        raw_text = resp.content.decode("utf-8-sig")
        data = json.loads(raw_text)
        log("    MNIS API: JSON parsed successfully (BOM stripped)")
    except json.JSONDecodeError:
        # The MNIS API may return XML even when JSON is requested.
        # Fall back to parsing it as XML.
        log("    MNIS API returned non-JSON, attempting XML parse...", "WARN")
        try:
            import xml.etree.ElementTree as ET
            raw_text = resp.content.decode("utf-8-sig")
            root = ET.fromstring(raw_text)

            # XML namespace used by MNIS
            ns = {"m": "http://schemas.datacontract.org/2004/07/ParlamentUK.Model.Core"}
            # Try without namespace too
            commons_lookup = {}

            # Find all Member elements (try with and without namespace)
            member_elems = root.findall(".//Member") or root.findall(f".//m:Member", ns)
            if not member_elems:
                # The root might itself be <Members> with <Member> children
                member_elems = list(root)

            for mem_elem in member_elems:
                # Get Member_Id from attribute
                member_id_str = mem_elem.get("Member_Id", "")
                if not member_id_str:
                    # Try child element
                    id_elem = mem_elem.find("Member_Id") or mem_elem.find(f"m:Member_Id", ns)
                    member_id_str = id_elem.text if id_elem is not None else ""
                try:
                    member_id = int(member_id_str)
                except (ValueError, TypeError):
                    continue

                # Find HouseMembership elements
                for hm in mem_elem.iter():
                    if "HouseMembership" in hm.tag and hm.tag.endswith("HouseMembership"):
                        house_elem = hm.find("House") or hm.find(f"m:House", ns)
                        house_text = ""
                        if house_elem is not None:
                            house_text = house_elem.text or ""
                            # Could be nested
                            if not house_text:
                                for child in house_elem:
                                    if child.text:
                                        house_text = child.text
                                        break

                        if "commons" not in house_text.lower():
                            continue

                        start_elem = hm.find("StartDate") or hm.find(f"m:StartDate", ns)
                        end_elem = hm.find("EndDate") or hm.find(f"m:EndDate", ns)
                        start = (start_elem.text or "")[:10] if start_elem is not None else ""
                        end = (end_elem.text or "")[:10] if end_elem is not None else ""

                        if member_id not in commons_lookup:
                            commons_lookup[member_id] = {"start": start, "end": end}
                        elif start > commons_lookup[member_id]["start"]:
                            commons_lookup[member_id] = {"start": start, "end": end}

            if commons_lookup:
                log(f"    MNIS XML parse: found {len(commons_lookup)} members with Commons service")
                return commons_lookup
            else:
                log("    MNIS XML parse found no members — falling back to Synopsis API", "WARN")
                return None

        except Exception as e2:
            log(f"    MNIS API: both JSON and XML parsing failed: {e2}", "WARN")
            return None

    # Parse the MNIS response to extract Commons membership periods.
    # The MNIS JSON structure nests Members -> Member (list) -> each has
    # HouseMemberships -> HouseMembership (list) with House, StartDate, EndDate.
    commons_lookup = {}
    members_list = data if isinstance(data, list) else data.get("Members", {}).get("Member", [])
    if isinstance(members_list, dict):
        members_list = [members_list]

    for member in members_list:
        try:
            member_id = int(member.get("@Member_Id", 0))
        except (ValueError, TypeError):
            continue

        # Walk their house memberships to find Commons ones
        hm_block = member.get("HouseMemberships", {})
        hm_list = hm_block.get("HouseMembership", []) if isinstance(hm_block, dict) else []
        if isinstance(hm_list, dict):
            hm_list = [hm_list]

        for hm in hm_list:
            house_name = ""
            if isinstance(hm.get("House"), dict):
                house_name = hm["House"].get("#text", "")
            elif isinstance(hm.get("House"), str):
                house_name = hm["House"]

            if "commons" not in house_name.lower():
                continue

            # MNIS dates can be plain strings ("2024-05-30T00:00:00")
            # OR dicts ({"#text": "2024-05-30T00:00:00"} or {"@xsi:nil": "true"})
            raw_start = hm.get("StartDate", "")
            raw_end = hm.get("EndDate", "")

            if isinstance(raw_start, dict):
                raw_start = raw_start.get("#text", "")
            if isinstance(raw_end, dict):
                raw_end = raw_end.get("#text", "")

            start = str(raw_start)[:10] if raw_start else ""
            end = str(raw_end)[:10] if raw_end else ""

            # Keep the latest Commons membership for each member
            if member_id not in commons_lookup:
                commons_lookup[member_id] = {"start": start, "end": end}
            else:
                # If this stint is more recent, update
                if start > commons_lookup[member_id]["start"]:
                    commons_lookup[member_id] = {"start": start, "end": end}

    log(f"    MNIS API returned {len(commons_lookup)} members with Commons service")
    return commons_lookup


def _check_prior_commons_via_synopsis(member_id):
    """
    Fallback: Use the Synopsis API endpoint to check for Commons service.
    Looks for patterns like "MP for Maidenhead from 1997 to 2024" or
    "elected to the House of Commons".

    Returns (had_commons: bool, start: str, end: str)
    """
    synopsis_url = f"{API_BASE}/Members/{member_id}/Synopsis"
    resp = safe_get(synopsis_url)
    if not resp:
        return False, "", ""

    try:
        syn_data = resp.json()
        synopsis_html = syn_data.get("value", "")
        if not synopsis_html:
            return False, "", ""

        soup = BeautifulSoup(synopsis_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
    except Exception:
        return False, "", ""

    # Pattern: "MP for X from YYYY" or "MP for X (YYYY-YYYY)"
    # or "Member of Parliament for X from YYYY to YYYY"
    patterns = [
        # "Member of Parliament for Maidenhead from 1997 to 2024"
        r"Member\s+of\s+Parliament\s+for\s+.+?\s+from\s+(\d{4})\s+to\s+(\d{4})",
        # "MP for Maidenhead 1997-2024" / "MP for Maidenhead (1997–2024)"
        r"MP\s+for\s+.+?\s+[\(]?(\d{4})\s*[-–]\s*(\d{4})",
        # "elected MP for Maidenhead in 1997" (end date unknown)
        r"elected\s+(?:as\s+)?(?:the\s+)?MP\s+for\s+.+?\s+in\s+(\d{4})",
        # "sat in the House of Commons from YYYY to YYYY"
        r"(?:sat|served)\s+in\s+the\s+House\s+of\s+Commons\s+from\s+(\d{4})\s+to\s+(\d{4})",
        # "was MP for X from 1997 to 2024"
        r"was\s+(?:the\s+)?MP\s+for\s+.+?\s+from\s+(\d{4})\s+to\s+(\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            start = groups[0] + "-01-01" if len(groups) >= 1 else ""
            end = groups[1] + "-12-31" if len(groups) >= 2 else ""
            return True, start, end

    return False, "", ""


def filter_by_date_range(commons_items, lords_items):
    """
    Build a filtered list of members who served in the Commons between
    1990 and 2024.

    Phase A: For House=1 members, use latestHouseMembership dates directly.
    Phase B: For House=2 members, use the MNIS API (with Synopsis API
             fallback) to identify former MPs and recover their Commons dates.
             ONLY include Lords members with confirmed Commons service dates.
    """
    filtered = []

    # ── Phase A: Direct Commons members ──────────────────────────
    log("  Phase A: Filtering Commons members by date range...")
    commons_count = 0
    for item in commons_items:
        val = item.get("value", {})
        membership = val.get("latestHouseMembership", {})

        if membership.get("house") != 1:
            continue

        start_str = membership.get("membershipStartDate")
        end_str = membership.get("membershipEndDate")

        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "")) if start_str else None
        except (ValueError, AttributeError):
            start_dt = None

        try:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "")) if end_str else None
        except (ValueError, AttributeError):
            end_dt = None

        if start_dt and start_dt > DATE_CEIL:
            continue
        if end_dt and end_dt < DATE_FLOOR:
            continue

        filtered.append(val)
        commons_count += 1

    log(f"    {commons_count} Commons members passed date filter")

    # ── Phase B: Check Lords members for prior Commons service ───
    log("  Phase B: Checking Lords members for prior Commons service...")

    # Step 1: Try the MNIS API for a bulk lookup
    mnis_lookup = _fetch_all_former_commons_from_mnis()

    lords_as_former_mps = 0
    lords_skipped_no_evidence = 0
    lords_skipped_outside_dates = 0

    for item in lords_items:
        val = item.get("value", {})
        member_id = val.get("id")

        commons_start = ""
        commons_end = ""
        had_commons = False

        # Try MNIS lookup first (fast, no extra API calls)
        if mnis_lookup and member_id in mnis_lookup:
            had_commons = True
            commons_start = mnis_lookup[member_id]["start"]
            commons_end = mnis_lookup[member_id]["end"]
        elif mnis_lookup is None:
            # MNIS unavailable — fall back to Synopsis API (slow, per-member)
            had_commons, commons_start, commons_end = _check_prior_commons_via_synopsis(member_id)
            time.sleep(RATE_LIMIT_DELAY)

        if not had_commons:
            lords_skipped_no_evidence += 1
            continue

        # STRICT date check — require confirmed dates that overlap 1990-2024.
        # Do NOT give "benefit of the doubt" for empty dates.
        try:
            start_dt = datetime.fromisoformat(commons_start) if commons_start and len(commons_start) >= 4 else None
        except (ValueError, AttributeError):
            start_dt = None

        try:
            end_dt = datetime.fromisoformat(commons_end) if commons_end and len(commons_end) >= 4 else None
        except (ValueError, AttributeError):
            end_dt = None

        # If we have no parseable dates at all, skip — we can't confirm overlap
        if start_dt is None and end_dt is None:
            lords_skipped_no_evidence += 1
            continue

        if start_dt and start_dt > DATE_CEIL:
            lords_skipped_outside_dates += 1
            continue
        if end_dt and end_dt < DATE_FLOOR:
            lords_skipped_outside_dates += 1
            continue

        # Confirmed former MP with overlapping dates
        val["_commons_start_recovered"] = commons_start
        val["_commons_end_recovered"] = commons_end
        val["_was_former_mp"] = True

        filtered.append(val)
        lords_as_former_mps += 1
        display_name = val.get("nameDisplayAs", "?")
        log(f"    FOUND former MP in Lords: {display_name} (id={member_id}, Commons {commons_start}–{commons_end})")

    log(f"    Lords checked: {len(lords_items)}")
    log(f"    Former MPs found with confirmed dates: {lords_as_former_mps}")
    log(f"    Skipped (no Commons evidence): {lords_skipped_no_evidence}")
    log(f"    Skipped (outside 1990-2024): {lords_skipped_outside_dates}")
    log(f"Stage 1 filtering: {len(filtered)} total MPs served in Commons between 1990 and 2024")
    log(f"  ({commons_count} from Commons query + {lords_as_former_mps} recovered from Lords)")
    return filtered


# ─── Stage 2: Normalize names ────────────────────────────────────────

def normalize_name(raw_name):
    """
    Strip titles, honorifics, post-nominals, and clean up whitespace.
    Returns a clean 'Firstname Lastname' string.
    """
    name = raw_name.strip()
    for pattern in TITLES_TO_STRIP:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    # Remove stray punctuation left behind
    name = re.sub(r"[,.]", " ", name)
    # Collapse multiple spaces
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def is_peerage_name(name_display, name_full_title):
    """Detect if a name looks like a peerage/titled name rather than a personal name."""
    for pattern in PEERAGE_INDICATORS:
        if re.search(pattern, name_display, re.IGNORECASE):
            return True
        if re.search(pattern, name_full_title, re.IGNORECASE):
            return True
    # Check if the normalized form still contains " of " (place-based title)
    normalized = normalize_name(name_display)
    if re.search(r"\bof\s+[A-Z]", normalized):
        return True
    # Flag single-word normalized names (likely a place name like "Aberdare")
    if len(normalized.split()) <= 1 and normalized:
        return True
    return False


def build_records(members):
    """
    Build structured records from raw API data and normalize names.
    Returns a list of dicts and a list of members needing biography resolution.
    """
    records = []
    needs_bio = []

    for m in members:
        member_id = m.get("id")
        name_display = m.get("nameDisplayAs", "")
        name_list_as = m.get("nameListAs", "")
        name_full = m.get("nameFullTitle", "")
        party = m.get("latestParty", {}).get("name", "")
        gender = m.get("gender", "")
        membership = m.get("latestHouseMembership", {})
        constituency = membership.get("membershipFrom", "")

        # For former MPs now in the Lords, prefer the recovered Commons dates
        if m.get("_was_former_mp"):
            start_date = m.get("_commons_start_recovered", "")
            end_date = m.get("_commons_end_recovered", "")
        else:
            start_date = membership.get("membershipStartDate", "")
            end_date = membership.get("membershipEndDate", "")

        # Normalize the display name
        normalized = normalize_name(name_display)

        # Check if this looks like a peerage name needing resolution.
        # All members recovered from the Lords are force-flagged — they
        # inherently have peerage names (e.g. "Baroness May of Maidenhead").
        peerage_flag = m.get("_was_former_mp", False) or is_peerage_name(name_display, name_full)

        record = {
            "id": member_id,
            "name_api_display": name_display,
            "name_api_list": name_list_as,
            "name_api_full_title": name_full,
            "name_normalized": normalized,
            "name_resolved": "",  # filled in Stage 3 if needed
            "name_final": normalized,  # will be overwritten if resolved
            "resolution_method": "normalized",
            "party": party,
            "gender": gender,
            "constituency": constituency,
            "membership_start": start_date[:10] if start_date else "",
            "membership_end": end_date[:10] if end_date else "",
        }
        records.append(record)

        if peerage_flag:
            needs_bio.append(record)

    log(f"Stage 2 complete: {len(records)} records normalized")
    log(f"  -> {len(needs_bio)} members flagged as titled/peerage names needing biography resolution")
    return records, needs_bio


# ─── Stage 3: Resolve titled names via biography pages ───────────────

def fetch_biography_name(member_id):
    """
    Fetch the Parliament biography page for a member and use BeautifulSoup
    to extract their real name.

    The Parliament website at members.parliament.uk/member/{id}/career
    typically contains text like:
      "Her name is Theresa Mary May"
      "His name is ..."
    or in the page header/metadata.

    We also try the Synopsis API endpoint as a fallback.
    """
    real_name = None

    # Strategy 1: Try the Synopsis API endpoint (returns HTML snippet)
    synopsis_url = f"{API_BASE}/Members/{member_id}/Synopsis"
    resp = safe_get(synopsis_url)
    if resp:
        try:
            syn_data = resp.json()
            # The synopsis value contains HTML biography text
            synopsis_html = syn_data.get("value", "")
            if synopsis_html:
                soup = BeautifulSoup(synopsis_html, "html.parser")
                bio_text = soup.get_text(separator=" ", strip=True)
                name = extract_name_from_bio(bio_text)
                if name:
                    return name, "synopsis_api"
        except (json.JSONDecodeError, AttributeError):
            pass

    time.sleep(BIO_RATE_LIMIT)

    # Strategy 2: Scrape the career page from the Parliament website
    career_url = f"{BIOGRAPHY_URL}/{member_id}/career"
    resp = safe_get(career_url, headers={"Accept": "text/html"})
    if resp and resp.status_code == 200:
        soup = BeautifulSoup(resp.text, "html.parser")

        # Look for the "Her/His name is X" pattern in any text on the page
        page_text = soup.get_text(separator=" ", strip=True)
        name = extract_name_from_bio(page_text)
        if name:
            return name, "career_page"

        # Look in meta tags or structured data
        for meta in soup.find_all("meta"):
            content = meta.get("content", "")
            if "name is" in content.lower():
                name = extract_name_from_bio(content)
                if name:
                    return name, "meta_tag"

    time.sleep(BIO_RATE_LIMIT)

    # Strategy 3: Try the main member overview page
    overview_url = f"{BIOGRAPHY_URL}/{member_id}"
    resp = safe_get(overview_url, headers={"Accept": "text/html"})
    if resp and resp.status_code == 200:
        soup = BeautifulSoup(resp.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)
        name = extract_name_from_bio(page_text)
        if name:
            return name, "overview_page"

    return None, "unresolved"


def extract_name_from_bio(text):
    """
    Parse biography text to find the real name using common patterns found
    on Parliament biography pages:

      "Her name is Theresa Mary May"
      "His name is David William Donald Cameron"
      "born Theresa Mary Brasier"
      "Full name: Edward Miliband"

    Returns the extracted personal name or None.
    """
    if not text:
        return None

    # Decode any HTML entities
    text = html.unescape(text)

    patterns = [
        # "Her/His name is Firstname Lastname"
        r"(?:Her|His|Their)\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        # "born Firstname Middlename Lastname"
        r"\bborn\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        # "Full name: Firstname Lastname"
        r"[Ff]ull\s+name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        # "whose real name is ..."
        r"real\s+name\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        # "also known as Firstname Lastname"
        r"(?:also\s+)?known\s+as\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            # Sanity check: should be 2-4 words, no weird characters
            words = candidate.split()
            if 2 <= len(words) <= 5:
                # Filter out false positives that are place names or titles
                if not any(w.lower() in ("of", "the", "and", "upon") for w in words):
                    return candidate

    return None


def resolve_titled_names(needs_bio):
    """
    For each member flagged as having a titled/peerage name, attempt to
    resolve their real personal name by scraping biography pages.
    """
    if not needs_bio:
        log("Stage 3: No titled names to resolve")
        return 0, 0

    log(f"Stage 3: Resolving {len(needs_bio)} titled/peerage names via biography pages...")
    resolved_count = 0
    failed_count = 0

    for i, record in enumerate(needs_bio, 1):
        member_id = record["id"]
        display_name = record["name_api_display"]

        real_name, method = fetch_biography_name(member_id)

        if real_name:
            record["name_resolved"] = real_name
            record["name_final"] = real_name
            record["resolution_method"] = method
            resolved_count += 1
            log(f"  [{i}/{len(needs_bio)}] Resolved: '{display_name}' -> '{real_name}' (via {method})")
        else:
            # Fall back to normalized version of the full title
            record["name_resolved"] = ""
            record["resolution_method"] = "unresolved"
            failed_count += 1
            log(f"  [{i}/{len(needs_bio)}] Unresolved: '{display_name}' -- keeping normalized: '{record['name_normalized']}'", "WARN")

        if i % 10 == 0:
            log(f"  ...processed {i}/{len(needs_bio)} biography lookups")

        time.sleep(BIO_RATE_LIMIT)

    log(f"Stage 3 complete: {resolved_count} resolved, {failed_count} unresolved")
    return resolved_count, failed_count


# ─── Stage 4: Deduplicate, drop middle names, and write CSV ─────────

def drop_middle_names(name):
    """
    Reduce a full name to first name + last name only.
      'Theresa Mary May'                -> 'Theresa May'
      'David William Donald Cameron'    -> 'David Cameron'
      'Jacob Rees-Mogg'                -> 'Jacob Rees-Mogg'  (2 parts, unchanged)
      'Diane Abbott'                   -> 'Diane Abbott'     (2 parts, unchanged)
    """
    parts = name.strip().split()
    if len(parts) <= 2:
        return name.strip()
    return f"{parts[0]} {parts[-1]}"


def deduplicate(records):
    """
    Some MPs may appear multiple times (e.g., re-elected after a gap).
    Keep the record with the latest membership start date per member ID.
    Then strip middle names from name_final.
    """
    by_id = {}
    for r in records:
        mid = r["id"]
        if mid not in by_id:
            by_id[mid] = r
        else:
            # Keep the one with the later start date
            existing_start = by_id[mid].get("membership_start", "")
            new_start = r.get("membership_start", "")
            if new_start > existing_start:
                by_id[mid] = r

    # Strip middle names from the final name
    for r in by_id.values():
        r["name_final"] = drop_middle_names(r.get("name_final", ""))

    deduped = sorted(by_id.values(), key=lambda x: x.get("name_final", ""))
    log(f"Deduplication: {len(records)} records -> {len(deduped)} unique MPs")
    log(f"  (middle names stripped from all final names)")
    return deduped


def write_csv(records, filename):
    """Write final gazetteer to CSV."""
    fieldnames = [
        "id",
        "name_final",
        "name_api_display",
        "name_api_full_title",
        "name_normalized",
        "name_resolved",
        "resolution_method",
        "party",
        "gender",
        "constituency",
        "membership_start",
        "membership_end",
    ]

    filepath = Path(filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    log(f"Stage 4 complete: Wrote {len(records)} records to {filepath.resolve()}")
    return filepath


# ─── Main Pipeline ───────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("MP Gazetteer Builder -- UK House of Commons 1990-2024")
    log("=" * 60)

    # Stage 1: Fetch all members from BOTH houses, then filter
    commons_items, lords_items = fetch_all_members()
    if not commons_items and not lords_items:
        log("No members returned from API. Check your network.", "ERROR")
        sys.exit(1)
    filtered = filter_by_date_range(commons_items, lords_items)

    # Stage 2: Normalize names and flag titled/peerage names
    records, needs_bio = build_records(filtered)

    # Print some sample records before resolution
    log("")
    log("--- Sample normalized records (first 5) ---")
    for r in records[:5]:
        log(f"  {r['name_api_display']:40s} -> {r['name_normalized']}")

    log("")
    log("--- Sample titled names flagged for resolution ---")
    for r in needs_bio[:10]:
        log(f"  {r['name_api_display']:40s}  (id={r['id']})")

    # Stage 3: Resolve titled/peerage names via biography scraping
    resolved, unresolved = resolve_titled_names(needs_bio)

    # Stage 4: Deduplicate and write CSV
    final_records = deduplicate(records)
    outpath = write_csv(final_records, OUTPUT_FILE)

    # ─── Summary ─────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Raw API records fetched:          {len(commons_items) + len(lords_items)}")
    log(f"    - from Commons query (House=1): {len(commons_items)}")
    log(f"    - from Lords query (House=2):   {len(lords_items)}")
    log(f"  After date filtering (1990-2024): {len(filtered)}")
    log(f"  After normalization:              {len(records)}")
    log(f"  Titled names flagged:             {len(needs_bio)}")
    log(f"  Successfully resolved via bio:    {resolved}")
    log(f"  Unresolved (kept normalized):     {unresolved}")
    log(f"  Final unique MPs in CSV:          {len(final_records)}")
    log(f"  Output file: {outpath.resolve()}")
    log("=" * 60)


if __name__ == "__main__":
    main()