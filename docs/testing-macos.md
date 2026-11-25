# Testing edgerpy on a Mac

Thank you for testing. This builds edgerpy on your Mac, runs its tests, and produces the files we need
back. It takes about 15 minutes, most of it the one-off download of the build tools (about 1 GB), and
changes nothing outside the edgerpy folder and `~/.cache/edgerpy-build-env`.

## What you need

- A Mac with Apple Silicon (M1 or later) or Intel, macOS 11 or later.
- Miniforge (conda for conda-forge). If `conda --version` fails in Terminal, install it from
  https://conda-forge.org/download/ and open a new Terminal window.

You do not need Xcode, Homebrew, Python or R.

## Steps

In Terminal, in the edgerpy folder you were given (the one holding `pyproject.toml`):

```sh
bash tools/build_wheel.sh
```

The script prints five steps. The last line should be `done: .../dist`.

## What to send back

Please zip the `dist` folder and send it back, together with:

- the output of `uname -m` (`arm64` or `x86_64`) and `sw_vers -productVersion`;
- anything printed in red, or the last lines if the script stopped.

`dist/` holds the wheel (`edgerpy-<version>-cp313-cp313-macosx_*.whl`), the source package
(`edgerpy-<version>.tar.gz`) and `dist/logs/` (the build log and the test results).

## If something goes wrong

- `needs conda or mamba`: Miniforge is not installed or not on the PATH; open a new Terminal window
  after installing it.
- The environment step fails: delete `~/.cache/edgerpy-build-env` and run the script again.
- A build or test step fails: send `dist/logs/` anyway; the logs are what we need.

To remove everything afterwards: delete the edgerpy folder and `~/.cache/edgerpy-build-env`.
