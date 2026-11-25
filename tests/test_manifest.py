"""The vendored C and Fortran sources are unchanged since they were copied from upstream.

VENDORED.sha256 lists every vendored file with its sha256. The test runs from a source tree and is
skipped when the manifest is not there (e.g. on an installed package).
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "VENDORED.sha256"


def rows():
    if not MANIFEST.exists():
        return []
    return [line.split("\t") for line in MANIFEST.read_text().splitlines()
            if line and not line.startswith("#") and line.split("\t")[1] in ("copy", "patched", "excerpt", "generated")]


@pytest.mark.skipif(not MANIFEST.exists(), reason="not a source tree")
@pytest.mark.parametrize("row", rows(), ids=lambda r: r[0])
def test_vendored_file(row):
    path, copy_sha = row[0], row[7]
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == copy_sha, f"{path} changed since vendoring"
