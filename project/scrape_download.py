import json
import re
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

SOURCES = "sources_50.jsonl"
OUT_INDEX = "scrape_index.jsonl"
ARTIFACTS_DIR = Path("artifacts")

TIMEOUT = 30
SLEEP_BETWEEN = 0.5

MAX_HTML_BYTES = 2_000_000
MAX_PDFS_PER_TRIAL = 2
MAX_TOTAL_DOWNLOADS_PER_TRIAL = 6

UA = "Mozilla/5.0 (compatible; BEA-pilot/0.1; +https://example.invalid)"

PDF_RE = re.compile(r"\.pdf(\?|#|$)", re.IGNORECASE)

def safe_name(s: str) -> str:
    s = s.strip().replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)[:140]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def fetch(url: str):
    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    return r.url, ctype, r.content

def is_probably_pdf(ctype: str, final_url: str) -> bool:
    if "application/pdf" in ctype:
        return True
    return bool(PDF_RE.search(final_url))

def is_probably_html(ctype: str) -> bool:
    return "text/html" in ctype or "application/xhtml" in ctype

def discover_pdf_links(html: bytes, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        absu = urljoin(base_url, href)
        if PDF_RE.search(absu):
            links.append(absu)
    # de-dupe preserve
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def write_index_line(obj):
    with open(OUT_INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    # clear index
    Path(OUT_INDEX).write_text("", encoding="utf-8")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCES, "r", encoding="utf-8") as f:
        trials = [json.loads(line) for line in f if line.strip()]

    for t in trials:
        rct_id = t["rct_id"]
        urls = t.get("urls", [])

        trial_dir = ARTIFACTS_DIR / safe_name(rct_id)
        fetched_dir = trial_dir / "fetched"
        discovered_dir = trial_dir / "discovered"
        fetched_dir.mkdir(parents=True, exist_ok=True)
        discovered_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        pdfs_downloaded = 0

        def attempt(url, kind_hint="seed"):
            nonlocal downloaded, pdfs_downloaded
            if downloaded >= MAX_TOTAL_DOWNLOADS_PER_TRIAL:
                return

            started = time.time()
            try:
                final_url, ctype, content = fetch(url)
                elapsed = round(time.time() - started, 3)
                h = sha256_bytes(content)

                # Choose file extension
                if is_probably_pdf(ctype, final_url):
                    ext = "pdf"
                    pdfs_downloaded += 1
                elif is_probably_html(ctype):
                    ext = "html"
                else:
                    # try infer from path
                    path = urlparse(final_url).path
                    ext = (Path(path).suffix.lstrip(".") or "bin")[:8]

                fname = f"{kind_hint}_{downloaded:02d}_{safe_name(Path(urlparse(final_url).path).name or 'download')}.{ext}"
                outpath = fetched_dir / fname
                outpath.write_bytes(content)

                write_index_line({
                    "rct_id": rct_id,
                    "url": url,
                    "final_url": final_url,
                    "content_type": ctype,
                    "bytes": len(content),
                    "sha256": h,
                    "saved_to": str(outpath),
                    "ok": True,
                    "elapsed_s": elapsed,
                })

                downloaded += 1

                # If HTML, discover PDFs and try a couple
                if ext == "html" and len(content) <= MAX_HTML_BYTES and pdfs_downloaded < MAX_PDFS_PER_TRIAL:
                    pdf_links = discover_pdf_links(content, final_url)
                    # store discovered list
                    (discovered_dir / f"pdf_links_{downloaded:02d}.txt").write_text(
                        "\n".join(pdf_links), encoding="utf-8"
                    )
                    for pdf_url in pdf_links[: max(0, MAX_PDFS_PER_TRIAL - pdfs_downloaded)]:
                        if pdfs_downloaded >= MAX_PDFS_PER_TRIAL:
                            break
                        attempt(pdf_url, kind_hint="pdf")
                return

            except Exception as e:
                elapsed = round(time.time() - started, 3)
                write_index_line({
                    "rct_id": rct_id,
                    "url": url,
                    "ok": False,
                    "error": str(e),
                    "elapsed_s": elapsed,
                })

        # Try seed URLs first
        for u in urls:
            if downloaded >= MAX_TOTAL_DOWNLOADS_PER_TRIAL:
                break
            if pdfs_downloaded >= MAX_PDFS_PER_TRIAL and downloaded >= 2:
                # once we got some PDFs, don't overfetch
                pass
            attempt(u, kind_hint="seed")
            time.sleep(SLEEP_BETWEEN)

    print(f"Done. See {OUT_INDEX} and artifacts/")

if __name__ == "__main__":
    main()

