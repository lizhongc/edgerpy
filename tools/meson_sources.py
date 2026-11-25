"""Print the files of one compile group listed in VENDORED.sha256 (used by meson.build)."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
group = sys.argv[1]
for line in (root / "VENDORED.sha256").read_text().splitlines():
    if line and not line.startswith("#"):
        f = line.split("\t")
        if f[1] != "block" and f[2] == group:
            print(f[0])
