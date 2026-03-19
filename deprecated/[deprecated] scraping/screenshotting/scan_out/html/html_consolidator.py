#!/usr/bin/env python3

import json
from pathlib import Path


def html_folder_to_json_list(folder="."):
    folder_path = Path(folder)

    html_files = sorted(
        [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in {".html", ".htm"}]
    )

    html_list = []
    for file_path in html_files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html_list.append(f.read())

    output_path = folder_path / "html_list.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(html_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(html_list)} HTML files to {output_path}")


if __name__ == "__main__":
    html_folder_to_json_list()
