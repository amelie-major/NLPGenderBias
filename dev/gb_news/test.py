from typing import List

def list_child_sitemaps(sitemap_index_url: str) -> List[str]:
    """Return <loc> links from a sitemapindex."""
    xml = get(sitemap_index_url).content
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
    # Try the main index first
    idx = list_child_sitemaps("https://www.gbnews.com/sitemap.xml")
    numbered = [u for u in idx if "/feeds/sitemaps/sitemap_" in u and u.endswith(".xml")]
    print(f"✅ Found {len(numbered)} numbered sitemap files referenced by sitemap.xml")
    return len(numbered)

print(f"🔍 Counting numbered sitemaps...")
num_sitemaps = count_numbered_sitemaps()
print(f"📊 Total numbered sitemaps: {num_sitemaps}")