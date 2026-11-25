"""Every public name has a help page: numpydoc sections, parameters matching the signature, no R syntax."""

import inspect
import re

import pytest

import edgerpy as ep

FUNCTIONS = [n for n in ep.__all__ if not inspect.isclass(getattr(ep, n))]
CLASSES = [n for n in ep.__all__ if inspect.isclass(getattr(ep, n))]


def doc(name):
    d = inspect.getdoc(getattr(ep, name))
    assert d and d.strip(), f"{name}: no docstring"
    return d


def section(d, title):
    m = re.search(rf"^{title}\n-+\n(.*?)(?=^\w[\w ]*\n-+\n|\Z)", d, flags=re.M | re.S)
    return m.group(1) if m else None


@pytest.mark.parametrize("name", FUNCTIONS)
def test_function_doc(name):
    d = doc(name)
    params = section(d, "Parameters")
    assert section(d, "Returns") is not None, f"{name}: no Returns section"
    sig = [p.name for p in inspect.signature(getattr(ep, name)).parameters.values()
           if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    documented = re.findall(r"^(\w+) :", params or "", flags=re.M)
    assert documented == sig, f"{name}: documented {documented}, signature {sig}"


@pytest.mark.parametrize("name", FUNCTIONS + CLASSES)
def test_doc_text(name):
    d = doc(name)
    assert "adapted for edgerpy" in " ".join(d.split()), f"{name}: no credit line"
    long = [line for line in d.splitlines() if len(line) > 100]
    assert not long, f"{name}: lines over 100 characters: {long[:2]}"
    r_syntax = re.findall(r"<-|\bTRUE\b|\bFALSE\b|\bNULL\b|\w\$\w", d)
    assert not r_syntax, f"{name}: R syntax left: {r_syntax}"
