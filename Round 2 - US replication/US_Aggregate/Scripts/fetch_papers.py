"""
fetch_papers.py

Step 3 of the aggregate pipeline.

For each Tier 1 study in Data/tier_classified.jsonl, attempts to download
the paper PDF from the primary_paper_url and saves it to Data/papers/{seq_id}.pdf.

Failures are logged to Data/paper_fetch_log.jsonl so you can see which studies
need manual retrieval.

Run from this script's directory:
    python fetch_papers.py [--all-tiers] [--limit N]
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).resolve().parents[1] / "Data"
INPUT     = DATA_DIR / "Pipeline" / "tier_classified.jsonl"
PAPERS_DIR = DATA_DIR / "Papers"
LOG_PATH  = DATA_DIR / "Caches" / "paper_fetch_log.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}
TIMEOUT   = 20   # seconds per request
DELAY     = 1.0  # seconds between requests (be polite)


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def try_download(url: str, dest: Path) -> tuple[bool, str]:
    """Attempt to download url to dest. Returns (success, message)."""
    if not url or url.strip() in ("", "None"):
        return False, "no_url"

    # Skip clearly non-PDF landing pages we know will fail
    skip_domains = ["jstor.org", "springer.com/article", "tandfonline.com"]
    if any(d in url for d in skip_domains):
        return False, f"likely_paywalled: {url[:80]}"

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()

        # Accept if content looks like a PDF
        if data[:4] == b"%PDF" or "pdf" in content_type.lower():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True, f"ok ({len(data)//1024} KB)"

        # Sometimes we get HTML (journal landing page / paywall)
        return False, f"non_pdf_response: content_type={content_type[:60]}"

    except urllib.error.HTTPError as e:
        return False, f"http_{e.code}: {url[:80]}"
    except urllib.error.URLError as e:
        return False, f"url_error: {e.reason}"
    except Exception as e:
        return False, f"error: {type(e).__name__}: {str(e)[:80]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-tiers", action="store_true",
                        help="Fetch papers for all tiers, not just Tier 1")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process first N studies only (0=all)")
    args = parser.parse_args()

    records = []
    with open(INPUT) as f:
        for line in f:
            records.append(json.loads(line))

    if not args.all_tiers:
        records = [r for r in records if r.get("llm_tier") == 1]
        print(f"Tier 1 studies: {len(records)}")
    else:
        print(f"All studies: {len(records)}")

    if args.limit:
        records = records[:args.limit]

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    log = []
    success = 0
    skip    = 0

    for i, rec in enumerate(records):
        seq_id = rec["seq_id"]
        title  = rec.get("title", "")[:60]
        dest   = PAPERS_DIR / f"{seq_id}.pdf"

        if dest.exists():
            print(f"[{i+1}/{len(records)}] seq={seq_id} SKIP (already downloaded) {title}")
            skip += 1
            log.append({"seq_id": seq_id, "status": "already_exists", "url": rec.get("primary_paper_url","")})
            continue

        url = rec.get("primary_paper_url", "")
        ok, msg = try_download(url, dest)

        status = "ok" if ok else "fail"
        if ok:
            success += 1
        print(f"[{i+1}/{len(records)}] seq={seq_id} {status.upper()} {msg} | {title}")
        log.append({"seq_id": seq_id, "status": status, "message": msg, "url": url})

        time.sleep(DELAY)

    # Write log
    with open(LOG_PATH, "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    total = len(records)
    print(f"\nDone. {success} downloaded, {skip} skipped (existing), "
          f"{total - success - skip} failed out of {total}")
    print(f"Log → {LOG_PATH}")
    print(f"PDFs → {PAPERS_DIR}/")


if __name__ == "__main__":
    main()
