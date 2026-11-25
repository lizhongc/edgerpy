"""Make the source distribution edgerpy-<version>.tar.gz without git.

meson-python makes an sdist with `meson dist`, which needs a git checkout. An sdist is only the source
tree under <name>-<version>/ plus PKG-INFO, the core metadata, which is the same as the wheel's METADATA.
So this script takes PKG-INFO from a wheel built from the same tree.

    python tools/make_sdist.py SOURCE_DIR WHEEL OUT_DIR
"""

import io
import sys
import tarfile
import zipfile
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", "dist", "build", ".ipynb_checkpoints"}


def main():
    src, wheel, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    with zipfile.ZipFile(wheel) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        pkg_info = z.read(meta)
    fields = dict(line.split(": ", 1) for line in pkg_info.decode().splitlines()[:20] if ": " in line)
    name, version = fields["Name"], fields["Version"]
    top = f"{name}-{version}"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{top}.tar.gz"
    files = sorted(p for p in src.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.relative_to(src).parts))

    def clean(t):
        t.uid = t.gid = 0
        t.uname = t.gname = ""
        return t

    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for p in files:
            tar.add(p, arcname=f"{top}/{p.relative_to(src).as_posix()}", filter=clean)
        info = tarfile.TarInfo(f"{top}/PKG-INFO")
        info.size, info.mode = len(pkg_info), 0o644
        tar.addfile(clean(info), io.BytesIO(pkg_info))
    print(f"{target} ({len(files) + 1} files)")


if __name__ == "__main__":
    main()
