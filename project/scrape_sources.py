import json
from pathlib import Path
from urllib.parse import urlparse

INPUT = "trials_sampled_50.json"
OUTPUT = "sources_50.jsonl"

def is_url(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    try:
        u = urlparse(s.strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def uniq_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def main():
    trials = json.loads(Path(INPUT).read_text(encoding="utf-8"))

    lines = []
    for _, t in trials.items():
        rct_id = t.get("RCT ID") or t.get("RCT_ID") or "UNKNOWN"
        title = t.get("Title", "")

        urls = []

        # Papers: list of dicts with URL
        papers = t.get("Papers")
        if isinstance(papers, list):
            for p in papers:
                if isinstance(p, dict) and is_url(p.get("URL", "")):
                    urls.append(p["URL"].strip())

        # Public / program files
        for k in ["Public Data URL", "Program Files URL"]:
            v = t.get(k)
            if is_url(v or ""):
                urls.append(v.strip())

        # External Link(s) sometimes {Link, Description} or list
        ext = t.get("External Link(s)")
        if isinstance(ext, dict):
            link = ext.get("Link", "")
            if is_url(link):
                urls.append(link.strip())
        elif isinstance(ext, list):
            for e in ext:
                if isinstance(e, dict) and is_url(e.get("Link", "")):
                    urls.append(e["Link"].strip())

        urls = uniq_preserve(urls)

        lines.append({
            "rct_id": rct_id,
            "title": title,
            "urls": urls,
        })

    out = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n"
    Path(OUTPUT).write_text(out, encoding="utf-8")
    print(f"Wrote {len(lines)} trials to {OUTPUT}")

if __name__ == "__main__":
    main()

