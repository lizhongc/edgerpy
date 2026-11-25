"""catchSalmon (Gibbs samples, Gencode names) and catchRSEM on small synthetic output, against R's results."""

import numpy as np

import edgerpy as ep
from _check import DATA, load, same

FIX = DATA / "fixtures"


def test_catch():
    d = load("CATCH")
    for case, y in [("S1", ep.catchSalmon(str(FIX / "salmon_gc"), verbose=False)),
                    ("R1", ep.catchRSEM(str(FIX / "rsem_gc"), verbose=False))]:
        same(y["counts"].to_numpy(), d[f"out/{case}/counts"], f"{case} counts")
        same(np.asarray(y["counts"].index, dtype=str), d[f"out/{case}/counts/@rownames"], f"{case} names")
        for col in ["Overdispersion", "Length"]:
            same(y["genes"][col].to_numpy(), d[f"out/{case}/genes/{col}"], f"{case} {col}")
