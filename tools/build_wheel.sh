#!/usr/bin/env bash
# Build edgerpy's release files on Linux or macOS, with conda-forge's compilers (no Docker, no Homebrew),
# and test them.
#
#   bash tools/build_wheel.sh [--env DIR] [--out DIR] [--bitwise]
#
# Run from the edgerpy source folder (the one holding pyproject.toml). Needs conda or mamba
# (Miniforge: https://conda-forge.org/download/).
#   1. creates (once) a conda-forge build environment in --env (default: ~/.cache/edgerpy-build-env):
#      Python 3.13, the compilers (gcc/gfortran on Linux with the glibc 2.28 sysroot; clang/gfortran and
#      llvm-openmp on macOS), meson-python, auditwheel (Linux) or delocate (macOS), pytest, twine;
#   2. builds the wheel and repairs it: auditwheel -> manylinux_2_28 (Linux); delocate bundles libgfortran
#      and libomp (macOS);
#   3. makes the sdist (tools/make_sdist.py) and checks that a wheel builds from it;
#   4. installs the repaired wheel in the build environment and runs the tests (tests/);
#   5. twine check.
# The files go to --out (default: dist/). The log of each step is in --out/logs/.
# --bitwise runs the tests with EDGERPY_TEST_BITWISE=1 (identical bits to R; expected on Linux x86_64 only).

set -euo pipefail

SRC=$(pwd)
ENV_DIR=${HOME}/.cache/edgerpy-build-env
OUT=$SRC/dist
BITWISE=0
while [ $# -gt 0 ]; do
  case $1 in
    --env) ENV_DIR=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --bitwise) BITWISE=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -f "$SRC/pyproject.toml" ] && [ -d "$SRC/src/edgerpy" ] || { echo "run from the edgerpy source folder" >&2; exit 2; }
mkdir -p "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
LOG=$OUT/logs

OS=$(uname -s)
case $OS in
  Linux)  EXTRA="sysroot_linux-64=2.28 auditwheel patchelf" ;;
  Darwin) EXTRA="llvm-openmp delocate" ;;
  *) echo "edgerpy builds on Linux and macOS only" >&2; exit 2 ;;
esac

CONDA=$(command -v mamba || command -v conda || true)
[ -n "$CONDA" ] || { echo "needs conda or mamba (Miniforge)" >&2; exit 2; }

echo "== 1. build environment: $ENV_DIR"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$CONDA" create -y -p "$ENV_DIR" -c conda-forge --override-channels \
    python=3.13 numpy pandas matplotlib h5py pyarrow pytest pip meson-python meson ninja twine \
    c-compiler fortran-compiler $EXTRA > "$LOG/env.log" 2>&1 || { tail -20 "$LOG/env.log"; exit 1; }
fi
eval "$(conda shell.bash hook 2>/dev/null || "$CONDA" shell.bash hook)"
conda activate "$ENV_DIR"
# meson.build sets every compile flag (the reference build's); keep conda's include and link paths only
# (conda's CPPFLAGS also hold -DNDEBUG and -O2, which the R maths code must not get)
export CFLAGS="" FFLAGS="" FCFLAGS="" CXXFLAGS="" DEBUG_CFLAGS="" DEBUG_FFLAGS=""
export CPPFLAGS="-isystem $CONDA_PREFIX/include"
python -c "import sys; assert sys.version_info[:2] >= (3, 13)"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "== 2. wheel"
python -m pip wheel --no-build-isolation --no-deps -w "$WORK/raw" "$SRC" > "$LOG/wheel.log" 2>&1 \
  || { tail -30 "$LOG/wheel.log"; exit 1; }
RAW=$(ls "$WORK"/raw/edgerpy-*.whl)
rm -f "$OUT"/edgerpy-*.whl "$OUT"/edgerpy-*.tar.gz
if [ "$OS" = Linux ]; then
  auditwheel repair --plat "manylinux_2_28_$(uname -m)" -w "$OUT" "$RAW" > "$LOG/repair.log" 2>&1 \
    || { tail -30 "$LOG/repair.log"; exit 1; }
else
  delocate-wheel -v -w "$OUT" "$RAW" > "$LOG/repair.log" 2>&1 || { tail -30 "$LOG/repair.log"; exit 1; }
fi
WHEEL=$(ls "$OUT"/edgerpy-*.whl)
echo "   $WHEEL"

echo "== 3. sdist"
python "$SRC/tools/make_sdist.py" "$SRC" "$RAW" "$OUT"
SDIST=$(ls "$OUT"/edgerpy-*.tar.gz)
python -m pip wheel --no-build-isolation --no-deps -w "$WORK/from-sdist" "$SDIST" > "$LOG/sdist-wheel.log" 2>&1 \
  || { tail -30 "$LOG/sdist-wheel.log"; exit 1; }
echo "   builds a wheel: ok"

echo "== 4. tests (installed wheel)"
python -m pip install -q --force-reinstall --no-deps "$WHEEL"
(cd "$WORK" && python -c "import edgerpy, sys; print('   edgerpy', edgerpy.__version__, 'from', edgerpy.__file__)")
(cd "$WORK" && EDGERPY_TEST_BITWISE=$BITWISE python -m pytest -q -p no:cacheprovider "$SRC/tests" \
  2>&1 | tee "$LOG/tests.log" | tail -3)
grep -qE "[0-9]+ passed" "$LOG/tests.log" && ! grep -qE "[0-9]+ (failed|error)" "$LOG/tests.log" \
  || { echo "tests failed: see $LOG/tests.log" >&2; exit 1; }

echo "== 5. twine check"
twine check "$WHEEL" "$SDIST"

echo "done: $OUT"
