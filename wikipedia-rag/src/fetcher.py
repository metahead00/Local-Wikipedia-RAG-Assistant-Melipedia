"""
Wikipedia article fetcher using the Wikipedia API (requests only).
"""

import re
import time

import requests

WIKIPEDIA_API_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&prop=extracts&explaintext=1"
    "&titles={title}&format=json&redirects=1"
)


def clean_text(text: str) -> str:
    """
    Remove excessive whitespace, == Section == header markers,
    and lines shorter than 20 characters (usually navigation artifacts).
    Return clean plain text.
    """
    # Remove == Section == style headers (any nesting level)
    text = re.sub(r"={2,}[^=\n]+={2,}", "", text)

    # Collapse runs of blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip lines shorter than 20 characters (navigation artifacts, stray headers, etc.)
    lines = text.splitlines()
    lines = [line for line in lines if len(line.strip()) == 0 or len(line.strip()) >= 20]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def fetch_wikipedia_text(title: str) -> dict:
    """
    Fetch the plain-text content of a Wikipedia article.

    Returns:
    {
        "title": str,        # normalized title from Wikipedia
        "url": str,          # full Wikipedia URL
        "text": str,         # full plain text of article (cleaned)
        "success": bool
    }
    Returns {"success": False, "title": title, "error": str} on failure.
    """
    url = WIKIPEDIA_API_URL.format(title=requests.utils.quote(title, safe=""))

    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "melipedia-rag/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        time.sleep(0.3)
        return {"success": False, "title": title, "error": str(exc)}
    finally:
        # Rate-limit: always sleep after a request attempt
        pass

    time.sleep(0.3)

    try:
        data = response.json()
    except ValueError as exc:
        return {"success": False, "title": title, "error": f"JSON parse error: {exc}"}

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return {"success": False, "title": title, "error": "No pages in API response"}

    # The API returns a dict keyed by page ID; grab the first (and only) entry
    page = next(iter(pages.values()))

    # page ID -1 means the page does not exist
    if page.get("pageid") == -1 or "missing" in page:
        return {
            "success": False,
            "title": title,
            "error": f"Page not found: {title}",
        }

    # Disambiguation pages have no extract or a very short one; treat them as failures
    raw_text = page.get("extract", "")
    if not raw_text:
        return {
            "success": False,
            "title": title,
            "error": "Empty extract (possibly a disambiguation page)",
        }

    normalized_title = page.get("title", title)
    wiki_url = "https://en.wikipedia.org/wiki/{}".format(
        normalized_title.replace(" ", "_")
    )

    cleaned = clean_text(raw_text)

    return {
        "success": True,
        "title": normalized_title,
        "url": wiki_url,
        "text": cleaned,
    }


if __name__ == "__main__":
    result = fetch_wikipedia_text("Albert Einstein")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Text length: {len(result['text'])} chars")
    print(f"First 200 chars: {result['text'][:200]}")
    assert result["success"] == True
    assert len(result["text"]) > 1000

    fail = fetch_wikipedia_text("XYZXYZXYZ_nonexistent_page_123")
    assert fail["success"] == False
    print("fetcher.py smoke test PASSED")
