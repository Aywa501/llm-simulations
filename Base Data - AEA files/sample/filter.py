import json
import csv
import re
from urllib.parse import urlparse

INPUT_FILE = "trials_pruned.json"

# --------------------------
# Tier classification rules
# --------------------------

TIER1_DOMAINS = [
    "dataverse.harvard.edu",
    "dvn.iq.harvard.edu",
    "zenodo.org",
    "osf.io",
    "github.com",
    "figshare.com",
    "datadryad.org",
    "data.mendeley.com",
    "dropbox.com"
]

TIER2_DOMAINS = [
    "openicpsr.org",
    "icpsr.umich.edu",
    "sciencedirect.com",
    "journals.uchicago.edu",
    "onlinelibrary.wiley.com",
    "econometricsociety.org",
    "cambridge.org",
    "restud.oxfordjournals.org",
    "pubsonline.informs.org",
    "drive.google.com",
    "docs.google.com"
]

DOI_PREFIX_TIER1 = [
    "10.7910",   # Dataverse
    "10.5281",   # Zenodo
    "10.17605"   # OSF
]

DOI_PREFIX_TIER2 = [
    "10.3886"    # ICPSR
]

def extract_doi_prefix(url):
    match = re.search(r'10\.\d{4,9}/[^\s]+', url)
    if match:
        return match.group(0).split("/")[0]
    return None

def classify_url(url):
    if not url:
        return "OTHER"

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # Handle hdl (Dataverse handles)
    if "hdl.handle.net" in domain:
        return "TIER1"

    # Domain-based checks
    for d in TIER1_DOMAINS:
        if d in domain:
            return "TIER1"

    for d in TIER2_DOMAINS:
        if d in domain:
            return "TIER2"

    # AEA replication direct zip
    if "aeaweb.org" in domain and url.endswith(".zip"):
        return "TIER1"

    # DOI-based classification
    doi_prefix = extract_doi_prefix(url)
    if doi_prefix:
        if doi_prefix in DOI_PREFIX_TIER1:
            return "TIER1"
        if doi_prefix in DOI_PREFIX_TIER2:
            return "TIER2"

    return "OTHER"


# --------------------------
# Load data
# --------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

tier1 = {}
tier2 = {}
summary_rows = []

for key, record in data.items():
    url = record.get("Public Data URL", "").strip()
    rct_id = record.get("RCT ID")
    title = record.get("Title")

    tier = classify_url(url)

    summary_rows.append({
        "rct_id": rct_id,
        "title": title,
        "url": url,
        "tier": tier
    })

    if tier == "TIER1":
        tier1[key] = record
    elif tier == "TIER2":
        tier2[key] = record

# --------------------------
# Write outputs
# --------------------------

with open("tier1_urls.json", "w", encoding="utf-8") as f:
    json.dump(tier1, f, indent=2)

with open("tier2_urls.json", "w", encoding="utf-8") as f:
    json.dump(tier2, f, indent=2)

with open("tier_classification_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["rct_id", "title", "url", "tier"])
    writer.writeheader()
    writer.writerows(summary_rows)

print("Filtering complete.")
print(f"Tier 1 count: {len(tier1)}")
print(f"Tier 2 count: {len(tier2)}")
