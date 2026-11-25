"""catch* readers of quantifier output (R: catchSalmon.R, catchKallisto.R, catchRSEM.R, catchOarfish.R,
catchSalmonGene.R, catchKallistoGene.R, catchOarfishGene.R, splitGencodeTxNames.R; edgeR 4.99.16).

Each reader returns a DGEList (R's DGEList=TRUE). Text files are parsed with readr's own double parser
(_C.r_parse_dbl), so numbers are bitwise those R reads. kallisto output needs h5py and oarfish output needs
pyarrow; both are optional and imported only when used.
"""

import gzip
import json
import os
import re

import numpy as np
import pandas as pd

from . import _C, _r
from .DGEList import DGEList
from .helpers import frame

NA_REAL = np.array([0x7FF00000000007A2], dtype=np.uint64).view(np.float64)[0]   # R's NA_real_

_GENCODE_FIELDS = ["EnsT", "EnsG", "HavG", "HavT", "TxName", "GeneName", "AnnLength", "Type"]


# ------------------------------------------------------------------ small R equivalents

def _sub_version(x):
    """R sub("\\\\.[0-9]+", "", x): drop the first '.digits'."""
    return [re.sub(r"\.[0-9]+", "", s, count=1) for s in x]


def _is_gencode(name):
    return str(name).count("|") >= 8


def _strsplit_fixed(s, sep):
    """R strsplit(s, sep, fixed=TRUE)[[1]]: a match at the end does not give a trailing ""."""
    if s == "":
        return []
    parts = s.split(sep)
    if parts[-1] == "":
        parts.pop()
    return parts


def _basename(p):
    return os.path.basename(p.rstrip("/"))


def _dir(path, pattern=None):
    """R dir(path, pattern): visible names, sorted (C locale)."""
    names = sorted(n for n in os.listdir(path) if not n.startswith("."))
    if pattern is not None:
        names = [n for n in names if re.search(pattern, n)]
    return names


def _read_tsv(path, col_types):
    """readr::read_tsv(path, col_types) for 'c' (character), 'd' (double) and '_' (skip) columns.
    Returns {column name: list of str or float64 array}."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        lines = [ln for ln in f.read().splitlines() if ln != ""]
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    out = {}
    for k, t in enumerate(col_types):
        if t == "_":
            continue
        col = [r[k] if k < len(r) else "" for r in rows]
        out[header[k]] = col if t == "c" else _C.r_parse_dbl(col)
    return out


def _group_codes(group):
    """Codes and unique values in order of first appearance (R rowsum(reorder=FALSE) groups)."""
    codes, uniq = _r.factorize_first(np.asarray(group, dtype=object))
    return np.asarray(codes), [str(u) for u in uniq]


def _rowsum(x, group):
    return _r.rowsum(x, np.asarray(group, dtype=object))[0]


def _count_per_group(codes, ngroup):
    return np.bincount(codes, minlength=ngroup).astype(np.int32)


def _row_min_max(m):
    return np.min(m, axis=1), np.max(m, axis=1)


# ------------------------------------------------------------------ Gencode names

def splitGencodeTxNames(x, remove_version=True):
    """Split Gencode Transcript Names

    Split delimited transcript names provided by Gencode FASTA files into separate columns.

    Parameters
    ----------
    x : list of str
        Gencode transcript names.
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs.

    Returns
    -------
    DataFrame
        Of strings, one row per transcript name, with columns:

        ``EnsT``
            Ensembl transcript ID (without version number suffix if ``remove_version=True``).
        ``EnsG``
            Ensembl gene ID (without version number suffix if ``remove_version=True``).
        ``HavG``
            Havana gene ID.
        ``HavT``
            Havana transcript ID.
        ``TxName``
            Transcript name.
        ``GeneName``
            Gene name.
        ``AnnLength``
            Transcript length.
        ``Type``
            Biotype of transcript.

    Notes
    -----
    Gencode transcript names consist of different annotations pasted together with ``|`` delimiters.
    ``splitGencodeTxNames`` splits the annotations into eight columns. Names with fewer fields give
    fewer columns (or empty strings in a row that is shorter than the others); extra fields give
    further columns ``V9``, ``V10``, ...

    R documentation by Gordon Smyth; adapted for edgerpy.
    """
    s = [_strsplit_fixed(str(v), "|") for v in x]
    nc = max((len(p) for p in s), default=0)
    A = np.full((len(s), nc), "", dtype=object)
    for i, p in enumerate(s):
        A[i, :len(p)] = p
    if remove_version:
        for j in range(min(2, nc)):
            A[:, j] = _sub_version(A[:, j])
    names = _GENCODE_FIELDS[:nc] if nc <= 8 else _GENCODE_FIELDS + [f"V{k}" for k in range(9, nc + 1)]
    return pd.DataFrame(A, columns=names)


def _collapseGencodeTx2Gene(txann):
    """R collapseGencodeTx2Gene(): one row per gene (first-appearance order) with GeneName, TxTypes, NPCTx."""
    ensg = list(txann["EnsG"])
    codes, ensgu = _group_codes(ensg)
    first = np.unique(codes, return_index=True)[1]
    txtype = [re.sub(r"([^_]{3})[^_]*", r"\1", t) if "_" in t else t for t in txann["Type"]]
    types = [set() for _ in ensgu]
    for c, t in zip(codes, txtype):
        types[c].add(t)
    genetype = ["|".join(sorted(ts)) for ts in types]
    genetype = [g.replace("pro_cod_CDS_not_def", "CDS_not_def").replace("pro_cod_LoF", "LoF") for g in genetype]
    out = pd.DataFrame({"GeneName": list(txann["GeneName"].to_numpy()[first])}, index=ensgu)
    out["TxTypes"] = genetype
    out["NPCTx"] = _rowsum((txann["Type"] == "protein_coding").to_numpy(dtype=np.float64), ensg).reshape(-1)
    return out


# ------------------------------------------------------------------ shared steps

def _imputeEffectiveLengths(length, efflen):
    """R .imputeEffectiveLengths(): an effective length within 0.5 of the transcript length is reset to the
    row's minimum over the other samples (1 if there is none)."""
    length = np.asarray(length, dtype=np.float64)
    flag = (length[:, None] - efflen) < 0.5
    rows = np.flatnonzero(flag.any(axis=1))
    if rows.size:
        efflen = efflen.copy()
        sub = np.where(flag[rows], np.inf, efflen[rows])
        m = np.min(sub, axis=1)
        m[~np.isfinite(m)] = 1.0
        fr = flag[rows]
        efflen[rows] = np.where(fr, m[:, None], efflen[rows])
    return efflen


def _accumulate(overdisp, df, boot):
    """Add one sample's resamples (rows x resamples) to the overdispersion sums and df."""
    M = _r.rowMeans(boot)
    i = M > 0
    b = boot[i, :] - M[i][:, None]
    overdisp[i] = overdisp[i] + _r.rowSums(b * b) / M[i]
    df[i] = df[i] + boot.shape[1] - 1


def _moderate(overdisp, df):
    """Overdispersion per row with limited moderation (prior df 3) and the prior; NA without resamples."""
    i = df > 0
    if i.sum() > 0:
        overdisp[i] = overdisp[i] / df[i]
        dfmedian = _r.median(df[i].astype(np.float64))
        prior = _r.median(overdisp[i]) / _r.qf(0.5, dfmedian, 3.0)
        if prior < 1:
            prior = 1.0
        overdisp[i] = (3.0 * prior + df[i] * overdisp[i]) / (3.0 + df[i])
        overdisp = np.maximum(overdisp, 1.0)
        overdisp[~i] = prior
        return overdisp, float(prior)
    overdisp[:] = NA_REAL
    return overdisp, NA_REAL


def _length_stats(L):
    """exp(rowMeans(L)) and exp(max - min) per row of a log-length matrix, and the row means."""
    m = _r.rowMeans(L)
    lo, hi = _row_min_max(L)
    return _r.exp(m), _r.exp(hi - lo), m


def _divide(counts, overdisp, df, divide):
    if divide:
        if np.max(df) > 0:
            return counts / overdisp[:, None]
        print("No bootstrap or Gibbs samples, so counts not divided")
    return counts


def _output(counts, genes, prior, rtype, divide, offset_prior=None, other=None):
    y = DGEList(counts, genes=genes)
    y["overdispersion.prior"] = prior
    y["resample.type"] = list(rtype)
    y["divided.counts"] = bool(divide)
    if offset_prior is not None:
        y["offset.prior"] = offset_prior
    if other is not None:
        y["other"] = other
    return y


def _unpack_gencode(ann, remove_version, drop_annlength=True):
    """Append Gencode fields (all but EnsT) to the annotation and use EnsT as row names."""
    A = splitGencodeTxNames(list(ann.index), remove_version=remove_version)
    extra = A.iloc[:, 1:]
    if drop_annlength and "AnnLength" in extra.columns:
        extra = extra.drop(columns="AnnLength")
    out = pd.concat([ann.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    out.index = list(A["EnsT"])
    return out


def _tx_output(counts, ann, efflen, logl, overdisp, prior, df, rtype, names, remove_version, divide, offset_prior):
    """Transcript-level DGEList for Salmon, kallisto and RSEM."""
    ave, ratio, rm = _length_stats(logl)
    ann = ann.copy()
    ann["AveEffLen"] = ave
    ann["Max2MinEffLen"] = ratio
    ann["Overdispersion"] = overdisp
    rows = list(ann.index)
    if _is_gencode(rows[0]):
        ann = _unpack_gencode(ann, remove_version)
        rows = list(ann.index)
    counts = _divide(counts, overdisp, df, divide)
    op = frame(logl - rm[:, None], rows, names) if offset_prior else None
    return _output(frame(counts, rows, names), ann, prior, rtype, divide, op,
                   {"effective.length": frame(efflen, rows, names)})


def _salmon_paths(parent_dir, sample_dirs, marker):
    if sample_dirs is None:
        if parent_dir is None:
            parent_dir = "."
        sample_dirs = [d for d in _dir(parent_dir) if os.path.exists(os.path.join(parent_dir, d, marker))]
    sample_dirs = [str(d) for d in sample_dirs]
    paths = sample_dirs if parent_dir is None else [f"{parent_dir}/{d}" for d in sample_dirs]
    return paths


def _salmon_quant_file(path):
    q = f"{path}/quant.sf"
    if not os.path.exists(q):
        found = _dir(path, "^quant.sf")
        if not found:
            raise FileNotFoundError("quant.sf file not found at specified path")
        q = f"{path}/{found[0]}"
    return q


def _salmon_meta(path, j, rtype, verbose):
    with open(f"{path}/aux_info/meta_info.json") as f:
        meta = json.load(f)
    ntx = meta.get("num_targets", meta.get("num_valid_targets"))
    if ntx is None:
        raise ValueError("Can't find number of targets")
    nboot = meta.get("num_bootstraps")
    if nboot is None:
        raise ValueError("Can't find number of bootstraps")
    typ = meta.get("samp_type")
    if typ is None:
        # edgeR 4.99.16 stops here too ("replacement has length zero")
        raise ValueError("meta_info.json has no samp_type")
    rtype[j] = typ
    if verbose:
        print(f"{ntx} transcripts, {nboot} {typ} samples")
    return int(ntx), int(nboot)


def _salmon_boot(path, ntx, nboot):
    with gzip.open(f"{path}/aux_info/bootstrap/bootstraps.gz", "rb") as f:
        b = np.frombuffer(f.read(), dtype="<f8", count=ntx * nboot)
    return np.asfortranarray(b.reshape((ntx, nboot), order="F").astype(np.float64))


def _h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to read kallisto output: pip install 'edgerpy[kallisto]'") from None
    return h5py


def _read_kallisto(h5file):
    h5py = _h5py()
    if not os.path.exists(h5file):
        raise FileNotFoundError("abundance.h5 file not found at specified path")
    with h5py.File(h5file, "r") as f:
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in f["aux/ids"][()]]
        lengths = np.asarray(f["aux/lengths"][()]).reshape(-1)
        efflen = np.asarray(f["aux/eff_lengths"][()], dtype=np.float64).reshape(-1)
        nboot = int(np.asarray(f["aux/num_bootstrap"][()]).reshape(-1)[0])
        est = np.asarray(f["est_counts"][()], dtype=np.float64).reshape(-1)
        boot = None
        if nboot > 0:
            g = f["bootstrap"]
            boot = np.asfortranarray(np.column_stack([np.asarray(g[k][()], dtype=np.float64) for k in g.keys()]))
    return ids, lengths, efflen, nboot, est, boot


def _read_parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required to read oarfish output: pip install 'edgerpy[oarfish]'") from None
    t = pq.read_table(path)
    return np.asfortranarray(np.column_stack([c.to_numpy().astype(np.float64) for c in t.columns]))


def _oarfish_prefixes(parent_dir, prefixes):
    if prefixes is None:
        files = _dir("." if parent_dir is None else parent_dir, r"\.quant$")
        prefixes = [f[:-6] for f in files]
    else:
        prefixes = [str(p) for p in prefixes]
    if len(prefixes) < 1:
        raise FileNotFoundError("No oarfish output files")
    if parent_dir is not None:
        prefixes = [f"{parent_dir}/{p}" for p in prefixes]
    return prefixes


def _oarfish_nboot(prefix):
    with open(f"{prefix}.meta_info.json") as f:
        meta = json.load(f)
    nboot = meta.get("num_bootstraps")
    if nboot is None:
        raise ValueError("Can't find number of bootstraps")
    return int(nboot)


# ------------------------------------------------------------------ transcript level

def catchSalmon(parent_dir=None, sample_dirs=None, remove_version=True, divide=True, impute_eff_len=True,
                offset_prior=True, verbose=True):
    """Process Salmon Transcript Output

    Read transcript counts from Salmon output for a series of biological samples and use the bootstrap
    or Gibbs posterior samples to estimate the read-to-transcript-ambiguity for each transcript.

    Parameters
    ----------
    parent_dir : str, optional
        Parent directory containing the Salmon output directories. Defaults to the current directory
        when ``sample_dirs`` is not given either.
    sample_dirs : list of str, optional
        Names of, or paths to, the sample directories created by Salmon, one per RNA-seq sample. If
        ``parent_dir`` is also given, they are relative to it. By default, every subdirectory of
        ``parent_dir`` that contains an ``aux_info`` directory is read, in sorted (C locale) order.
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs of Gencode-style
        transcript names.
    divide : bool, default True
        Whether to divide the counts by the RTA overdispersions.
    impute_eff_len : bool, default True
        Whether to impute missing effective transcript lengths (see Notes).
    offset_prior : bool, default True
        Whether to include a prior offset matrix representing differences in effective transcript
        length.
    verbose : bool, default True
        Whether to print progress information as each sample is read.

    Returns
    -------
    DGEList
        With components:

        ``counts``
            DataFrame of transcript counts, one row per transcript and one column per sample (named
            by the sample directories).
        ``samples``
            DataFrame with ``group``, ``lib.size`` (column sums of ``counts``) and ``norm.factors``.
        ``genes``
            DataFrame of transcript information with columns ``Length``, ``AveEffLen``,
            ``Max2MinEffLen`` and ``Overdispersion``: transcript length, effective transcript length
            averaged over samples (geometric mean), ratio of largest to smallest effective length over
            the samples, and RTA overdispersion.
        ``overdispersion.prior``
            Median overdispersion, used to moderate the transcript-wise overdispersions (NaN when no
            sample has resamples).
        ``resample.type``
            List giving the type of resampling (``"bootstrap"`` or ``"gibbs"``) for each sample.
        ``divided.counts``
            Whether the counts were to be divided by the RTA overdispersions (the ``divide`` argument).
        ``offset.prior``
            DataFrame of prior offsets representing the relative differences in effective lengths for
            each transcript on the log scale (only if ``offset_prior=True``). Rows add to zero.
        ``other``
            Dict with ``"effective.length"``, the DataFrame of effective transcript lengths.

        If Gencode-style composite transcript names are detected, then the row names are reset to
        plain Ensembl transcript IDs (``EnsT``) and ``genes`` also has the columns ``EnsG`` (Ensembl
        gene ID), ``HavG`` (Havana gene ID), ``HavT`` (Havana transcript ID), ``TxName`` (transcript
        name), ``GeneName`` (gene name) and ``Type`` (type of transcript).

    Notes
    -----
    *Reading transcript counts.* ``catchSalmon`` assumes that Salmon (Patro et al 2017; Zakeri et al
    2017) has been run to estimate transcript counts for one or more RNA samples and that either
    bootstrap resamples or Gibbs posterior samples are included in the Salmon output. ``catchSalmon``
    automatically detects whether bootstrap or Gibbs samples are available. The number of technical
    resamples does not need to be the same for each RNA sample, but the resamples should always be of
    the same type, either bootstrap or Gibbs. We recommend that at least 200 bootstrap or Gibbs
    resamples are generated in total across all the RNA samples (Baldoni et al 2024b).

    *RTA overdispersion.* The function reads the transcript counts and uses the technical (bootstrap
    or Gibbs) resamples to estimate an overdispersion parameter for each transcript. The
    overdispersion represents the variance inflation that occurs from ambiguity in assigning sequence
    reads to transcripts, a phenomenon that is called read to transcript ambiguity (RTA)
    overdispersion by Baldoni et al (2024ab). Transcripts that overlap other transcripts and have
    greater quantification uncertainty will have larger RTA overdispersions. The RTA overdispersions
    are greater than or equal to 1, with 1 representing no variance inflation.

    *Divided counts.* To assess differential transcript expression, the transcript counts can be
    divided by the overdispersion parameters, after which the divided counts can be input into
    standard differential expression pipelines designed for gene-level counts (Baldoni et al 2024a).
    The edgeR quasi-likelihood pipeline has been found to perform well with the divided counts
    (Baldoni et al 2024ab). The divided counts behave much like negative binomial counts and show the
    same mean-variance trends as gene-level RNA-seq counts. While bootstrap and Gibbs resamples both
    perform well, Baldoni et al (2024b) show that Gibbs resamples are computationally faster than
    bootstrap resamples and give slightly better performance in the downstream differential
    expression analyses. If no sample has resamples, the counts are not divided and a message says so.

    *Imputing missing transcript lengths.* For paired-end RNA-seq, Salmon uses sample-specific
    empirical fragment distributions to estimate the effective transcript lengths. For very short
    transcripts, there may be too few entries in the empirical distribution to estimate the
    conditional fragment length. In such cases, Salmon falls back to setting the effective transcript
    length equal to the actual length even though the effective transcript length should in principle
    be much shorter. If ``impute_eff_len`` is ``True``, then any effective transcript length equal to
    the transcript's actual length is reset to the minimum effective transcript length for that
    transcript across all the samples in the series.

    *Gencode imbedded annotation.* Gencode packs transcript and gene annotation into composite
    transcript IDs, with fields separated by vertical bar ``|`` characters. ``catchSalmon`` detects
    Gencode-style transcript names and unpacks the annotation fields into separate annotation columns
    using ``splitGencodeTxNames``.

    ``quant.sf`` is parsed with readr's number parser, as R does, so every value is the one R reads.
    A missing ``quant.sf`` raises ``FileNotFoundError``; a ``meta_info.json`` without the number of
    targets, the number of bootstraps or ``samp_type`` raises ``ValueError`` (edgeR stops too when
    ``samp_type`` is missing).

    R documentation by Gordon Smyth and Pedro Baldoni; adapted for edgerpy.
    """
    paths = _salmon_paths(parent_dir, sample_dirs, "aux_info")
    ns = len(paths)
    rtype = [""] * ns
    for j, p in enumerate(paths):
        if verbose:
            print(f"Reading {_basename(p)}, ", end="")
        qfile = _salmon_quant_file(p)
        ntx, nboot = _salmon_meta(p, j, rtype, verbose)
        if j == 0:
            counts = np.zeros((ntx, ns), order="F")
            efflen = np.zeros((ntx, ns), order="F")
            df = np.zeros(ntx, dtype=np.int64)
            overdisp = np.zeros(ntx)
            q1 = _read_tsv(qfile, "cdd_d")
            counts[:, 0] = q1["NumReads"]
            efflen[:, 0] = q1["EffectiveLength"]
        else:
            q = _read_tsv(qfile, "__d_d")
            counts[:, j] = q["NumReads"]
            efflen[:, j] = q["EffectiveLength"]
        if nboot > 0:
            _accumulate(overdisp, df, _salmon_boot(p, ntx, nboot))
    if impute_eff_len:
        efflen = _imputeEffectiveLengths(q1["Length"], efflen)
    overdisp, prior = _moderate(overdisp, df)
    ann = pd.DataFrame({"Length": q1["Length"]}, index=q1["Name"])
    return _tx_output(counts, ann, efflen, _r.log(efflen), overdisp, prior, df, rtype,
                      [_basename(p) for p in paths], remove_version, divide, offset_prior)


def catchKallisto(parent_dir=None, sample_dirs=None, remove_version=True, divide=True, impute_eff_len=True,
                  offset_prior=True, verbose=True):
    """Process kallisto Transcript Output

    Read transcript counts from kallisto output for a series of biological samples and use the
    bootstrap samples to estimate the read-to-transcript-ambiguity for each transcript.

    Parameters
    ----------
    parent_dir : str, optional
        Parent directory containing the kallisto output directories. Defaults to the current
        directory when ``sample_dirs`` is not given either.
    sample_dirs : list of str, optional
        Names of, or paths to, the sample directories created by kallisto, one per RNA-seq sample. If
        ``parent_dir`` is also given, they are relative to it. By default, every subdirectory of
        ``parent_dir`` that contains an ``abundance.h5`` file is read, in sorted (C locale) order.
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs of Gencode-style
        transcript names.
    divide : bool, default True
        Whether to divide the counts by the RTA overdispersions.
    impute_eff_len : bool, default True
        Whether to impute missing effective transcript lengths (see Notes).
    offset_prior : bool, default True
        Whether to include a prior offset matrix representing differences in effective transcript
        length.
    verbose : bool, default True
        Whether to print progress information as each sample is read.

    Returns
    -------
    DGEList
        With components:

        ``counts``
            DataFrame of transcript counts, one row per transcript and one column per sample (named
            by the sample directories).
        ``samples``
            DataFrame with ``group``, ``lib.size`` (column sums of ``counts``) and ``norm.factors``.
        ``genes``
            DataFrame of transcript information with columns ``Length``, ``AveEffLen``,
            ``Max2MinEffLen`` and ``Overdispersion``: transcript length, effective transcript length
            averaged over samples (geometric mean), ratio of largest to smallest effective length over
            the samples, and RTA overdispersion.
        ``overdispersion.prior``
            Median overdispersion, used to moderate the transcript-wise overdispersions (NaN when no
            sample has bootstrap samples).
        ``resample.type``
            List giving the type of resampling (``"bootstrap"``) for each sample.
        ``divided.counts``
            Whether the counts were to be divided by the RTA overdispersions (the ``divide`` argument).
        ``offset.prior``
            DataFrame of prior offsets representing the relative differences in effective lengths for
            each transcript on the log scale (only if ``offset_prior=True``). Rows add to zero.
        ``other``
            Dict with ``"effective.length"``, the DataFrame of effective transcript lengths.

        If Gencode-style composite transcript names are detected, then the row names are reset to
        plain Ensembl transcript IDs (``EnsT``) and ``genes`` also has the columns ``EnsG`` (Ensembl
        gene ID), ``HavG`` (Havana gene ID), ``HavT`` (Havana transcript ID), ``TxName`` (transcript
        name), ``GeneName`` (gene name) and ``Type`` (type of transcript).

    Notes
    -----
    *Reading transcript counts.* ``catchKallisto`` assumes that kallisto (Bray et al 2016) has been
    run to estimate transcript counts for one or more RNA samples and that bootstrap samples have also
    been generated. The number of bootstrap resamples does not need to be the same for each RNA
    sample. ``catchKallisto`` reads the HDF5 files ``abundance.h5`` output by kallisto with the
    optional package h5py (``pip install 'edgerpy[kallisto]'``); without it, ``ImportError`` is
    raised. A missing ``abundance.h5`` raises ``FileNotFoundError``.

    *RTA overdispersion.* The function reads the transcript counts and uses the bootstrap resamples
    to estimate an overdispersion parameter for each transcript. The overdispersion represents the
    variance inflation that occurs from ambiguity in assigning sequence reads to transcripts, a
    phenomenon that is called read to transcript ambiguity (RTA) overdispersion by Baldoni et al
    (2024ab). Transcripts that overlap other transcripts and have greater quantification uncertainty
    will have larger RTA overdispersions. The RTA overdispersions are greater than or equal to 1, with
    1 representing no variance inflation.

    *Divided counts.* To assess differential transcript expression, the transcript counts can be
    divided by the overdispersion parameters, after which the divided counts can be input into
    standard differential expression pipelines designed for gene-level counts (Baldoni et al 2024a).
    The edgeR quasi-likelihood pipeline has been found to perform well with the divided counts
    (Baldoni et al 2024ab). The divided counts behave much like negative binomial counts and show the
    same mean-variance trends as gene-level RNA-seq counts. If no sample has bootstrap samples, the
    counts are not divided and a message says so.

    *Imputing missing transcript lengths.* For paired-end RNA-seq, kallisto uses sample-specific
    empirical fragment distributions to estimate the effective transcript lengths. For very short
    transcripts, there may be too few entries in the empirical distribution to estimate the
    conditional fragment length. In such cases, kallisto falls back to setting the effective
    transcript length equal to the actual length even though the effective transcript length should in
    principle be much shorter. If ``impute_eff_len`` is ``True``, then any effective transcript length
    equal to the transcript's actual length is reset to the minimum effective transcript length for
    that transcript across all the samples in the series.

    *Gencode imbedded annotation.* Gencode packs transcript and gene annotation into composite
    transcript IDs, with fields separated by vertical bar ``|`` characters. ``catchKallisto`` detects
    Gencode-style transcript names and unpacks the annotation fields into separate annotation columns
    using ``splitGencodeTxNames``.

    R documentation by Gordon Smyth and Pedro Baldoni; adapted for edgerpy.
    """
    _h5py()
    paths = _salmon_paths(parent_dir, sample_dirs, "abundance.h5")
    ns = len(paths)
    for j, p in enumerate(paths):
        if verbose:
            print(f"Reading {_basename(p)}, ", end="")
        ids, lengths, eff, nboot, est, boot = _read_kallisto(f"{p}/abundance.h5")
        ntx = len(lengths)
        if verbose:
            print(f"{ntx} transcripts, {nboot} bootstraps")
        if j == 0:
            counts = np.zeros((ntx, ns), order="F")
            efflen = np.zeros((ntx, ns), order="F")
            df = np.zeros(ntx, dtype=np.int64)
            overdisp = np.zeros(ntx)
        counts[:, j] = est
        efflen[:, j] = eff
        if nboot > 0:
            _accumulate(overdisp, df, boot)
    if impute_eff_len:
        efflen = _imputeEffectiveLengths(lengths, efflen)
    overdisp, prior = _moderate(overdisp, df)
    ann = pd.DataFrame({"Length": lengths.astype(np.int32)}, index=ids)
    return _tx_output(counts, ann, efflen, _r.log(efflen), overdisp, prior, df, ["bootstrap"] * ns,
                      [_basename(p) for p in paths], remove_version, divide, offset_prior)


def _removeExt(x):
    """limma removeExt(x): drop the extension after the last '.' when every element has the same one."""
    if any("." not in s and s == "" for s in x):
        return list(x)
    ext = [re.sub(r"^(.*)\.(.*)$", r"\2", s) for s in x]
    if all(e == ext[0] for e in ext):
        return [re.sub(r"^(.*)\.(.*)$", r"\1", s) for s in x]
    return list(x)


def catchRSEM(parent_dir=None, files=None, ngibbs=100, remove_version=True, divide=True, offset_prior=True,
              verbose=True):
    """Process RSEM Transcript Output

    Read transcript counts from RSEM output for a series of biological samples and use the Gibbs
    posterior distribution samples to estimate the read-to-transcript-ambiguity for each transcript.

    Parameters
    ----------
    parent_dir : str, optional
        Directory containing the RSEM output files. Defaults to the current directory.
    files : list of str, optional
        The ``isoforms.results`` files output by RSEM, relative to ``parent_dir``, one per RNA-seq
        sample. By default, all files in ``parent_dir`` with names ending in ``.isoforms.results`` are
        read, in sorted (C locale) order.
    ngibbs : int or list of int, default 100
        Number of Gibbs samples used to generate the RSEM results files: one value for all files, or
        one per file (recycled).
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs of Gencode-style
        transcript names.
    divide : bool, default True
        Whether to divide the counts by the RTA overdispersions.
    offset_prior : bool, default True
        Whether to include a prior offset matrix representing differences in effective transcript
        length.
    verbose : bool, default True
        Whether to print progress information as each file is read.

    Returns
    -------
    DGEList
        With components:

        ``counts``
            DataFrame of transcript counts (RSEM's expected counts), one row per transcript and one
            column per sample (named by the files, without the ``.isoforms.results`` extensions).
        ``samples``
            DataFrame with ``group``, ``lib.size`` (column sums of ``counts``) and ``norm.factors``.
        ``genes``
            DataFrame of transcript information with columns ``Length``, ``AveEffLen``,
            ``Max2MinEffLen`` and ``Overdispersion``: transcript length, effective transcript length
            averaged over samples (geometric mean), ratio of largest to smallest effective length over
            the samples, and RTA overdispersion.
        ``overdispersion.prior``
            Median overdispersion, used to moderate the transcript-wise overdispersions (NaN when no
            sample has Gibbs samples).
        ``resample.type``
            List giving the type of resampling (``"gibbs"``) for each sample.
        ``divided.counts``
            Whether the counts were to be divided by the RTA overdispersions (the ``divide`` argument).
        ``offset.prior``
            DataFrame of prior offsets representing the relative differences in effective lengths for
            each transcript on the log scale (only if ``offset_prior=True``). Rows add to zero.
        ``other``
            Dict with ``"effective.length"``, the DataFrame of effective transcript lengths.

        If Gencode-style composite transcript names are detected, then the row names are reset to
        plain Ensembl transcript IDs (``EnsT``) and ``genes`` also has the columns ``EnsG`` (Ensembl
        gene ID), ``HavG`` (Havana gene ID), ``HavT`` (Havana transcript ID), ``TxName`` (transcript
        name), ``GeneName`` (gene name) and ``Type`` (type of transcript).

    Notes
    -----
    *Reading transcript counts.* ``catchRSEM`` reads ``*.isoforms.results`` files created by RSEM
    (Li and Dewey, 2011), which should contain means and standard deviations from Gibbs posterior
    samples as well as the estimated transcript counts. ``catchRSEM`` cannot detect the number of
    Gibbs samples performed for each RNA sample, so this must be specified via ``ngibbs``. A missing
    file raises ``FileNotFoundError``, and a file without the ``expected_count`` column raises
    ``ValueError``.

    *RTA overdispersion.* The function reads the transcript counts and uses the Gibbs posterior
    means and standard deviations to estimate an overdispersion parameter for each transcript. The
    overdispersion represents the variance inflation that occurs from ambiguity in assigning sequence
    reads to transcripts, a phenomenon that is called read to transcript ambiguity (RTA)
    overdispersion by Baldoni et al (2024ab). Transcripts that overlap other transcripts and have
    greater quantification uncertainty will have larger RTA overdispersions. The RTA overdispersions
    are greater than or equal to 1, with 1 representing no variance inflation.

    *Divided counts.* To assess differential transcript expression, the transcript counts can be
    divided by the overdispersion parameters, after which the divided counts can be input into
    standard differential expression pipelines designed for gene-level counts (Baldoni et al 2024a).
    The edgeR quasi-likelihood pipeline has been found to perform well with the divided counts
    (Baldoni et al 2024ab). The divided counts behave much like negative binomial counts and show the
    same mean-variance trends as gene-level RNA-seq counts. If ``ngibbs`` is zero for every file, the
    counts are not divided and a message says so.

    In ``AveEffLen``, ``Max2MinEffLen`` and ``offset.prior``, an effective length of zero counts as
    1e-8; ``other["effective.length"]`` keeps RSEM's values.

    *Gencode imbedded annotation.* Gencode packs transcript and gene annotation into composite
    transcript IDs, with fields separated by vertical bar ``|`` characters. ``catchRSEM`` detects
    Gencode-style transcript names and unpacks the annotation fields into separate annotation columns
    using ``splitGencodeTxNames``.

    The results files are parsed with readr's number parser, as R does, so every value is the one R
    reads.

    R documentation by Gordon Smyth and Pedro Baldoni; adapted for edgerpy.
    """
    if parent_dir is None:
        parent_dir = "."
    if files is None:
        files = _dir(parent_dir, r"\.isoforms\.results$")
    else:
        files = [str(f) for f in files]
    ns = len(files)
    if ns < 1:
        raise FileNotFoundError("No isoforms.results files")
    files = [f"{parent_dir}/{f}" for f in files]
    ngibbs = np.resize(np.asarray(ngibbs, dtype=np.float64).reshape(-1), ns)
    for j, f in enumerate(files):
        if verbose:
            print(f"Reading {f}, ", end="")
        if not os.path.exists(f):
            raise FileNotFoundError("isoforms.results file not found at specified path")
        if j == 0:
            q1 = _read_tsv(f, "c_ddd___dd___")
            if "expected_count" not in q1:
                raise ValueError("File doesn't contain expected_count column")
            ntx = len(q1["transcript_id"])
            counts = np.zeros((ntx, ns), order="F")
            efflen = np.zeros((ntx, ns), order="F")
            df = np.zeros(ntx)
            overdisp = np.zeros(ntx)
            q = q1
        else:
            q = _read_tsv(f, "___dd___dd___")
            if "expected_count" not in q:
                raise ValueError("File doesn't contain expected_count column")
        counts[:, j] = q["expected_count"]
        efflen[:, j] = q["effective_length"]
        M, S = q["posterior_mean_count"], q["posterior_standard_deviation_of_count"]
        ng = ngibbs[j]
        if verbose:
            print(f"{ntx} transcripts, {_r.num2str(ng)} Gibbs samples")
        if ng > 0:
            i = M > 0
            overdisp[i] = overdisp[i] + (ng - 1) * (S[i] * S[i]) / M[i]
            df[i] = df[i] + ng - 1
    logl = _r.log(efflen)
    logl[np.isinf(logl)] = _r.log(1e-8)
    overdisp, prior = _moderate(overdisp, df)
    names = _removeExt(_removeExt([_basename(f) for f in files]))
    ann = pd.DataFrame({"Length": q1["length"]}, index=q1["transcript_id"])
    return _tx_output(counts, ann, efflen, logl, overdisp, prior, df, ["gibbs"] * ns, names, remove_version,
                      divide, offset_prior)


def catchOarfish(parent_dir=None, prefixes=None, remove_version=True, divide=True, verbose=True):
    """Process Oarfish Transcript Output

    Read transcript counts from Oarfish output for a series of biological samples and use the
    bootstrap samples to estimate the read-to-transcript-ambiguity for each transcript.

    Parameters
    ----------
    parent_dir : str, optional
        Directory containing the Oarfish output files. Defaults to the current directory.
    prefixes : list of str, optional
        The sample-specific file prefixes created by Oarfish, one per RNA-seq sample; each may contain
        a directory separator. If ``parent_dir`` is given, they are relative to it. By default, the
        prefixes of all files in ``parent_dir`` with names ending in ``.quant`` are used, in sorted
        (C locale) order. For each prefix, ``<prefix>.quant``, ``<prefix>.meta_info.json`` and, when
        there are bootstrap samples, ``<prefix>.infreps.pq`` are read.
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs of Gencode-style
        transcript names.
    divide : bool, default True
        Whether to divide the counts by the RTA overdispersions.
    verbose : bool, default True
        Whether to print progress information as each sample is read.

    Returns
    -------
    DGEList
        With components:

        ``counts``
            DataFrame of transcript counts, one row per transcript and one column per sample (named
            by the prefixes, without directories).
        ``samples``
            DataFrame with ``group``, ``lib.size`` (column sums of ``counts``) and ``norm.factors``.
        ``genes``
            DataFrame of transcript information with columns ``Length`` (transcript length) and
            ``Overdispersion`` (RTA overdispersion).
        ``overdispersion.prior``
            Median overdispersion, used to moderate the transcript-wise overdispersions (NaN when no
            sample has bootstrap samples).
        ``resample.type``
            List giving the type of resampling (``"bootstrap"``) for each sample.
        ``divided.counts``
            Whether the counts were divided by the RTA overdispersions (the ``divide`` argument).

        There is no ``offset.prior`` and no ``other``: Oarfish gives no effective lengths.

        If Gencode-style composite transcript names are detected, then the row names are reset to
        plain Ensembl transcript IDs (``EnsT``) and ``genes`` also has the columns ``EnsG`` (Ensembl
        gene ID), ``HavG`` (Havana gene ID), ``HavT`` (Havana transcript ID), ``TxName`` (transcript
        name), ``GeneName`` (gene name), ``AnnLength`` (annotated transcript length) and ``Type``
        (type of transcript).

    Notes
    -----
    *Reading transcript counts.* ``catchOarfish`` assumes that Oarfish (Jousheghani et al 2025) has
    been run to estimate transcript counts for one or more RNA samples and that bootstrap resamples
    are included in the output. The bootstrap samples (Parquet files) are read with the optional
    package pyarrow (``pip install 'edgerpy[oarfish]'``); without it, ``ImportError`` is raised. No
    output files raise ``FileNotFoundError``, as does a missing ``.quant`` file.

    *RTA overdispersion.* The function reads the transcript counts and uses the bootstrap resamples
    to estimate an overdispersion parameter for each transcript. The overdispersion represents the
    variance inflation that occurs from ambiguity in assigning sequence reads to transcripts, a
    phenomenon that is called read to transcript ambiguity (RTA) overdispersion by Baldoni et al
    (2024ab). Transcripts that overlap other transcripts and have greater quantification uncertainty
    will have larger RTA overdispersions. The RTA overdispersions are greater than or equal to 1, with
    1 representing no variance inflation.

    *Divided counts.* To assess differential transcript expression, the transcript counts can be
    divided by the overdispersion parameters, after which the divided counts can be input into
    standard differential expression pipelines designed for gene-level counts (Baldoni et al 2024a).
    The edgeR quasi-likelihood pipeline has been found to perform well with the divided counts
    (Baldoni et al 2024ab). The divided counts behave much like negative binomial counts and show the
    same mean-variance trends as gene-level RNA-seq counts. As in edgeR, ``divide=True`` divides even
    when there are no bootstrap samples, which then gives counts of NaN; use ``divide=False`` there.

    *Gencode imbedded annotation.* Gencode packs transcript and gene annotation into composite
    transcript IDs, with fields separated by vertical bar ``|`` characters. ``catchOarfish`` detects
    Gencode-style transcript names and unpacks the annotation fields into separate annotation columns
    using ``splitGencodeTxNames``.

    The ``.quant`` files are parsed with readr's number parser, as R does, so every value is the one
    R reads.

    R documentation by Gordon Smyth and Pedro Baldoni; adapted for edgerpy.
    """
    prefixes = _oarfish_prefixes(parent_dir, prefixes)
    ns = len(prefixes)
    for j, pre in enumerate(prefixes):
        if verbose:
            print(f"Reading {pre}, ", end="")
        if not os.path.exists(f"{pre}.quant"):
            raise FileNotFoundError("quant file not found at specified path")
        nboot = _oarfish_nboot(pre)
        if verbose:
            print(f"{nboot} bootstraps")
        if j == 0:
            q = _read_tsv(f"{pre}.quant", "cdd")
            ntx = len(q["tname"])
            counts = np.zeros((ntx, ns), order="F")
            df = np.zeros(ntx, dtype=np.int64)
            overdisp = np.zeros(ntx)
            counts[:, 0] = q["num_reads"]
            ann = pd.DataFrame({"Length": q["len"]}, index=q["tname"])
        else:
            counts[:, j] = _read_tsv(f"{pre}.quant", "__d")["num_reads"]
        if nboot > 0:
            _accumulate(overdisp, df, _read_parquet(f"{pre}.infreps.pq"))
    overdisp, prior = _moderate(overdisp, df)
    ann["Overdispersion"] = overdisp
    rows = list(ann.index)
    if _is_gencode(rows[0]):
        ann = _unpack_gencode(ann, remove_version, drop_annlength=False)
        rows = list(ann.index)
    if divide:
        counts = counts / overdisp[:, None]
    return _output(frame(counts, rows, [_basename(p) for p in prefixes]), ann, prior, ["bootstrap"] * ns, divide)


# ------------------------------------------------------------------ gene level

def _tx2gene_map(txid, tx2gene, remove_version, verbose, strip_gene_version=True):
    """Gene ID of each transcript, from Gencode names (tx2gene None) or a tx2gene table. Returns (EnsG, GeneAnn)."""
    is_gc = _is_gencode(txid[0])
    if tx2gene is None:
        if not is_gc:
            raise ValueError("tx2gene not provided and row names do not appear to be from Gencode")
        if verbose:
            print("Summarizing to genewise counts using Gencode's imbedded annotation")
        txann = splitGencodeTxNames(txid, remove_version=remove_version)
        return list(txann["EnsG"]), _collapseGencodeTx2Gene(txann)
    t2g = pd.DataFrame(tx2gene)
    if t2g.shape[1] < 2:
        raise ValueError("tx2gene doesn't have two columns")
    c1 = [str(v) for v in t2g.iloc[:, 0]]
    c2 = [None if pd.isna(v) else str(v) for v in t2g.iloc[:, 1]]
    if is_gc and not _is_gencode(c1[0]):
        if verbose:
            print("Salmon tx names are Gencode-style but tx2gene tx names are not, so unpacking Salmon tx names")
        txid = [re.sub(r"[|].*", "", s, count=1) for s in txid]
    if remove_version:
        c1 = _sub_version(c1)
        if strip_gene_version:            # edgeR 4.99.16: catchKallistoGene leaves the gene IDs alone
            c2 = [None if v is None else re.sub(r"\.[0-9]+", "", v, count=1) for v in c2]
        txid = _sub_version(txid)
    pos = {}
    for k, v in enumerate(c1):
        pos.setdefault(v, k)
    m = [pos.get(v) for v in txid]
    if any(v is None for v in m):
        raise ValueError("Tx names not found in first column of tx2gene")
    ensg = [c2[k] for k in m]
    if any(v is None for v in ensg):
        raise ValueError("Missing gene IDs")
    if verbose:
        print("Summarizing to genewise counts using tx2gene")
    return ensg, None


def _max_by_group(x, codes, ngroup):
    out = np.full(ngroup, -np.inf)
    np.maximum.at(out, codes, np.asarray(x, dtype=np.float64))
    return out.astype(np.asarray(x).dtype)


def _gene_length(method, tpm, txlen, ensg, ensgu, ntx_per_gene, counts, tximport_len):
    """Gene effective lengths for gene.length = "moderate", "tximport" or "simple" (Salmon and kallisto).
    tximport_len: the per-tx length matrix the tximport method averages (effective tx lengths)."""
    if method == "moderate":
        m = _r.rowMeans(tpm)
        tpm2 = tpm + 1e-6 + (m / 100)[:, None]
        return _rowsum(tpm2 * txlen, ensg) / _rowsum(tpm2, ensg)
    if method == "tximport":
        ensg_a = np.asarray(ensg, dtype=object)
        allzero = np.flatnonzero(_r.rowSums(counts) == 0)
        if allzero.size:
            txz = np.flatnonzero(np.isin(ensg_a, np.asarray(ensgu, dtype=object)[allzero]))
            m = _r.rowMeans(tximport_len[txz, :])
            out = counts.copy()
            out[allzero, :] = (_rowsum(m, ensg_a[txz]) / ntx_per_gene[allzero][:, None])
            keep = np.setdiff1d(np.arange(len(ensg_a)), txz)
            rest = np.setdiff1d(np.arange(counts.shape[0]), allzero)
            with np.errstate(invalid="ignore", divide="ignore"):         # 0/0 is NaN, as in R; filled below
                out[rest, :] = (_rowsum(tpm[keep] * tximport_len[keep], ensg_a[keep]) /
                                _rowsum(tpm[keep], ensg_a[keep]))
            if np.isnan(out).any():
                m = _r.exp(_r.rowMeans(_r.log(out), na_rm=True))
                i = np.isnan(out)
                out[i] = np.broadcast_to(m[:, None], out.shape)[i]
            return out
        with np.errstate(invalid="ignore", divide="ignore"):
            return _rowsum(tpm * tximport_len, ensg) / _rowsum(tpm, ensg)
    if method == "simple":
        return _rowsum(txlen, ensg) / ntx_per_gene[:, None]
    raise ValueError("gene.length should be one of 'moderate', 'tximport', 'simple'")


def _gene_output(counts, genelen, tpm_gene, ensg, ensgu, codes, geneann, maxtxlen, overdisp, prior, df, rtype,
                 names, divide, offset_prior, range_col=True, divide_ok=None, divide_msg=None):
    L = _r.log(genelen)
    ave, ratio, m = _length_stats(L)
    ntx = _count_per_group(codes, len(ensgu))
    cols = {} if geneann is None else {c: geneann[c].to_numpy() for c in geneann.columns}
    cols["NTx"] = ntx
    cols["MaxTxLen"] = maxtxlen
    cols["AveEffLen"] = ave
    if range_col:
        cols["Max2MinEffLen"] = ratio
    cols["Overdispersion"] = overdisp
    genes = pd.DataFrame(cols, index=ensgu)
    if divide:
        ok = (np.max(df) > 0) if divide_ok is None else divide_ok
        if ok:
            counts = counts / overdisp[:, None]
        else:
            print(divide_msg or "No bootstrap or Gibbs samples, so counts not divided")
    op = frame(L - m[:, None], ensgu, names) if offset_prior else None
    other = {"effective.length": frame(genelen, ensgu, names), "tpm": frame(tpm_gene, ensgu, names)}
    return _output(frame(counts, ensgu, names), genes, prior, rtype, divide, op, other)


def _check_gene_length(gene_length):
    if gene_length not in ("moderate", "tximport", "simple"):
        raise ValueError("'gene_length' should be one of 'moderate', 'tximport', 'simple'")


def catchSalmonGene(parent_dir=None, sample_dirs=None, tx2gene=None, remove_version=True, divide=True,
                    impute_eff_len=True, offset_prior=True, gene_length="moderate", verbose=True):
    """Process Salmon Output to Produce Gene Counts

    Produce a gene count object from Salmon output for a series of biological samples, optionally with
    Gencode annotation.

    Parameters
    ----------
    parent_dir : str, optional
        Parent directory containing the Salmon output directories. Defaults to the current directory
        when ``sample_dirs`` is not given either.
    sample_dirs : list of str, optional
        Names of, or paths to, the sample directories created by Salmon, one per RNA-seq sample. If
        ``parent_dir`` is also given, they are relative to it. By default, every subdirectory of
        ``parent_dir`` that contains an ``aux_info`` directory is read, in sorted (C locale) order.
    tx2gene : DataFrame, optional
        Maps transcript IDs to gene IDs: the first column holds transcript IDs and the second the
        corresponding gene IDs. Not needed if Gencode annotation was used for the quantification.
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs.
    divide : bool, default True
        Whether to divide the gene counts by the RTA overdispersions.
    impute_eff_len : bool, default True
        Whether to impute missing effective transcript lengths (see Notes).
    offset_prior : bool, default True
        Whether to include a prior offset matrix representing differences in effective gene length.
    gene_length : {"moderate", "tximport", "simple"}, default "moderate"
        How to compute the effective gene length (see Notes).
    verbose : bool, default True
        Whether to print progress information as each sample is read.

    Returns
    -------
    DGEList
        With gene IDs as row names, in order of first appearance, and components:

        ``counts``
            DataFrame of gene counts, one row per gene and one column per sample (named by the
            sample directories).
        ``samples``
            DataFrame with ``group``, ``lib.size`` (column sums of ``counts``) and ``norm.factors``.
        ``genes``
            DataFrame of gene annotation: ``GeneName`` (gene name), ``TxTypes`` (unique transcript
            biotypes for that gene, pipe-separated) and ``NPCTx`` (number of protein-coding
            transcripts) if Gencode annotation is used; then ``NTx`` (number of transcripts),
            ``MaxTxLen`` (maximum transcript length), ``AveEffLen`` (effective gene length, averaged
            over samples), ``Max2MinEffLen`` (ratio of largest to smallest effective gene length
            across samples) and ``Overdispersion`` (genewise RTA overdispersion).
        ``overdispersion.prior``
            Median of the genewise overdispersions, used to moderate the overdispersion values (NaN
            when no sample has resamples).
        ``resample.type``
            List giving the resampling type (``"bootstrap"`` or ``"gibbs"``) for each sample.
        ``divided.counts``
            Whether the counts were to be divided by the RTA overdispersions (the ``divide`` argument).
        ``offset.prior``
            DataFrame of prior offsets representing the relative differences in effective lengths for
            each gene on the log scale (only if ``offset_prior=True``). Rows add to zero.
        ``other``
            Dict with ``"effective.length"``, the DataFrame of effective gene lengths, and ``"tpm"``,
            the DataFrame of genewise TPM values.

    Notes
    -----
    ``catchSalmonGene`` assumes that Salmon (Patro et al 2017; Zakeri et al 2017) has been run to
    estimate transcript counts for one or more RNA samples, and that either bootstrap resamples or
    Gibbs posterior samples are included in the Salmon output. The function reads the Salmon
    quantifications and summarizes them at the gene level. It automatically detects whether bootstrap
    or Gibbs samples are available.

    The transcript-level counts are collapsed to gene-level counts and RTA overdispersions are
    estimated at the gene level, using a method similar to Baldoni et al (2024ab) but applied to gene
    level counts instead of transcript counts. If no sample has resamples, the counts are not divided
    and a message says so.

    The function needs to know which gene each transcript belongs to. If Salmon was run with a Gencode
    transcript FASTA file, then ``catchSalmonGene`` detects that automatically. It then unpacks the
    Gencode transcript names using ``splitGencodeTxNames`` to obtain gene and transcript IDs.
    Otherwise, a DataFrame ``tx2gene`` needs to be provided to map transcript to gene IDs. If the
    transcript names are Gencode-style but those of ``tx2gene`` are not, the Ensembl transcript IDs
    are taken from the Gencode names. Without ``tx2gene``, names that are not Gencode-style raise
    ``ValueError``, as do transcripts missing from ``tx2gene`` and missing gene IDs.

    Average transcript lengths are computed for each gene and optionally used to create a prior offset
    matrix. The average lengths are computed by one of three methods, as specified by
    ``gene_length``: a simple average of the transcript lengths (``"simple"``), the method of
    Soneson et al (2016) (``"tximport"``), or a method that moderates the lengths weakly towards the
    gene average and, even more weakly, towards the simple average of transcript lengths
    (``"moderate"``). The moderated method is recommended.

    For paired-end RNA-seq, Salmon uses sample-specific empirical fragment distributions to estimate
    the effective transcript lengths. For very short transcripts, there may be too few entries in the
    empirical distribution to estimate the conditional fragment length. In such cases, Salmon falls
    back to setting the effective transcript length equal to the actual length even though the
    effective transcript length should in principle be much shorter. If ``impute_eff_len`` is
    ``True``, then any effective transcript length equal to the transcript's actual length is reset to
    the minimum effective transcript length for that transcript across all the samples in the series.

    ``quant.sf`` is parsed with readr's number parser, as R does, so every value is the one R reads.

    R documentation by Gordon Smyth and Pedro Baldoni; adapted for edgerpy.
    """
    _check_gene_length(gene_length)
    paths = _salmon_paths(parent_dir, sample_dirs, "aux_info")
    ns = len(paths)
    rtype = [""] * ns
    for j, p in enumerate(paths):
        if verbose:
            print(f"Reading {_basename(p)}, ", end="")
        qfile = _salmon_quant_file(p)
        ntx, nboot = _salmon_meta(p, j, rtype, verbose)
        if j == 0:
            q1 = _read_tsv(qfile, "cdddd")
            ensg, geneann = _tx2gene_map(q1["Name"], tx2gene, remove_version, verbose)
            codes, ensgu = _group_codes(ensg)
            ng = len(ensgu)
            ntx_per_gene = _count_per_group(codes, ng)
            counts = np.zeros((ng, ns), order="F")
            tpm = np.zeros((ntx, ns), order="F")
            efftx = np.zeros((ntx, ns), order="F")
            df = np.zeros(ng, dtype=np.int64)
            overdisp = np.zeros(ng)
            q = q1
        else:
            q = _read_tsv(qfile, "__ddd")
        counts[:, j] = _rowsum(q["NumReads"], ensg).reshape(-1)
        tpm[:, j] = q["TPM"]
        efftx[:, j] = q["EffectiveLength"]
        if nboot > 0:
            _accumulate(overdisp, df, np.asfortranarray(_rowsum(_salmon_boot(p, ntx, nboot), ensg)))
    maxtxlen = _max_by_group(q1["Length"], codes, ng)
    if impute_eff_len:
        efftx = _imputeEffectiveLengths(q1["Length"], efftx)
    genelen = _gene_length(gene_length, tpm, efftx, ensg, ensgu, ntx_per_gene, counts, efftx)
    overdisp, prior = _moderate(overdisp, df)
    return _gene_output(counts, genelen, _rowsum(tpm, ensg), ensg, ensgu, codes, geneann, maxtxlen, overdisp, prior,
                        df, rtype, [_basename(p) for p in paths], divide, offset_prior)


def catchKallistoGene(parent_dir=None, sample_dirs=None, tx2gene=None, remove_version=True, divide=True,
                      impute_eff_len=True, offset_prior=True, gene_length="moderate", verbose=True):
    """Process kallisto Output to Produce Gene Counts

    Produce a gene count object from kallisto output for a series of biological samples, optionally
    with Gencode annotation.

    Parameters
    ----------
    parent_dir : str, optional
        Parent directory containing the kallisto output directories. Defaults to the current
        directory when ``sample_dirs`` is not given either.
    sample_dirs : list of str, optional
        Names of, or paths to, the sample directories created by kallisto, one per RNA-seq sample. If
        ``parent_dir`` is also given, they are relative to it. By default, every subdirectory of
        ``parent_dir`` that contains an ``abundance.h5`` file is read, in sorted (C locale) order.
    tx2gene : DataFrame, optional
        Maps transcript IDs to gene IDs: the first column holds transcript IDs and the second the
        corresponding gene IDs. Not needed if Gencode annotation was used for the quantification.
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs. As in edgeR 4.99.16,
        the gene IDs of a ``tx2gene`` table keep their version numbers.
    divide : bool, default True
        Whether to divide the gene counts by the RTA overdispersions.
    impute_eff_len : bool, default True
        Whether to impute missing effective transcript lengths (see Notes).
    offset_prior : bool, default True
        Whether to include a prior offset matrix representing differences in effective gene length.
    gene_length : {"moderate", "tximport", "simple"}, default "moderate"
        How to compute the effective gene length (see Notes).
    verbose : bool, default True
        Whether to print progress information as each sample is read.

    Returns
    -------
    DGEList
        With gene IDs as row names, in order of first appearance, and components:

        ``counts``
            DataFrame of gene counts, one row per gene and one column per sample (named by the
            sample directories).
        ``samples``
            DataFrame with ``group``, ``lib.size`` (column sums of ``counts``) and ``norm.factors``.
        ``genes``
            DataFrame of gene annotation: ``GeneName`` (gene name), ``TxTypes`` (unique transcript
            biotypes for that gene, pipe-separated) and ``NPCTx`` (number of protein-coding
            transcripts) if Gencode annotation is used; then ``NTx`` (number of transcripts),
            ``MaxTxLen`` (maximum transcript length), ``AveEffLen`` (effective gene length, averaged
            over samples), ``Max2MinEffLen`` (ratio of largest to smallest effective gene length
            across samples) and ``Overdispersion`` (genewise RTA overdispersion).
        ``overdispersion.prior``
            Median of the genewise overdispersions, used to moderate the overdispersion values (NaN
            when no sample has bootstrap samples).
        ``resample.type``
            List giving the resampling type (``"bootstrap"``) for each sample.
        ``divided.counts``
            Whether the counts were to be divided by the RTA overdispersions (the ``divide`` argument).
        ``offset.prior``
            DataFrame of prior offsets representing the relative differences in effective lengths for
            each gene on the log scale (only if ``offset_prior=True``). Rows add to zero.
        ``other``
            Dict with ``"effective.length"``, the DataFrame of effective gene lengths, and ``"tpm"``,
            the DataFrame of genewise TPM values.

    Notes
    -----
    ``catchKallistoGene`` assumes that kallisto (Bray et al 2016) has been run to estimate transcript
    counts for one or more RNA samples and that bootstrap samples have also been generated. The number
    of bootstrap resamples does not need to be the same for each RNA sample. ``catchKallistoGene``
    reads the HDF5 files ``abundance.h5`` output by kallisto with the optional package h5py
    (``pip install 'edgerpy[kallisto]'``); without it, ``ImportError`` is raised. A missing
    ``abundance.h5`` raises ``FileNotFoundError``.

    The transcript-level counts are collapsed to gene-level counts and RTA overdispersions are
    estimated at the gene level, using a method similar to Baldoni et al (2024ab) but applied to gene
    level counts instead of transcript counts. If no sample has bootstrap samples, the counts are not
    divided and a message says so.

    The function needs to know which gene each transcript belongs to. If kallisto was run with a
    Gencode transcript FASTA file, then ``catchKallistoGene`` detects that automatically. It then
    unpacks the Gencode transcript names using ``splitGencodeTxNames`` to obtain gene and transcript
    IDs. Otherwise, a DataFrame ``tx2gene`` needs to be provided to map transcript to gene IDs. If the
    transcript names are Gencode-style but those of ``tx2gene`` are not, the Ensembl transcript IDs
    are taken from the Gencode names. Without ``tx2gene``, names that are not Gencode-style raise
    ``ValueError``, as do transcripts missing from ``tx2gene`` and missing gene IDs.

    Average transcript lengths are computed for each gene and optionally used to create a prior offset
    matrix. The average lengths are computed by one of three methods, as specified by
    ``gene_length``: a simple average of the transcript lengths (``"simple"``), the method of
    Soneson et al (2016) (``"tximport"``), or a method that moderates the lengths weakly towards the
    gene average and, even more weakly, towards the simple average of transcript lengths
    (``"moderate"``). The moderated method is recommended.

    For paired-end RNA-seq, kallisto uses sample-specific empirical fragment distributions to estimate
    the effective transcript lengths. For very short transcripts, there may be too few entries in the
    empirical distribution to estimate the conditional fragment length. In such cases, kallisto falls
    back to setting the effective transcript length equal to the actual length even though the
    effective transcript length should in principle be much shorter. If ``impute_eff_len`` is
    ``True``, then any effective transcript length equal to the transcript's actual length is reset to
    the minimum effective transcript length for that transcript across all the samples in the series.

    R documentation by Gordon Smyth and Pedro Baldoni; adapted for edgerpy.
    """
    _check_gene_length(gene_length)
    _h5py()
    paths = _salmon_paths(parent_dir, sample_dirs, "abundance.h5")
    ns = len(paths)
    for j, p in enumerate(paths):
        if verbose:
            print(f"Reading {_basename(p)}, ", end="")
        ids, lengths, eff, nboot, est, boot = _read_kallisto(f"{p}/abundance.h5")
        if verbose:
            print(f"{len(ids)} transcripts, {nboot} bootstraps")
        if j == 0:
            ntx = len(ids)
            txlen = lengths
            ensg, geneann = _tx2gene_map(ids, tx2gene, remove_version, verbose, strip_gene_version=False)
            codes, ensgu = _group_codes(ensg)
            ng = len(ensgu)
            ntx_per_gene = _count_per_group(codes, ng)
            counts = np.zeros((ng, ns), order="F")
            tpm = np.zeros((ntx, ns), order="F")
            efftx = np.zeros((ntx, ns), order="F")
            df = np.zeros(ng, dtype=np.int64)
            overdisp = np.zeros(ng)
        counts[:, j] = _rowsum(est, ensg).reshape(-1)
        efftx[:, j] = eff
        x = est / eff
        tpm[:, j] = 1e6 * x / _r.rsum(x)
        if nboot > 0:
            _accumulate(overdisp, df, np.asfortranarray(_rowsum(boot, ensg)))
    maxtxlen = _max_by_group(txlen, codes, ng)
    if impute_eff_len:
        efftx = _imputeEffectiveLengths(txlen, efftx)
    genelen = _gene_length(gene_length, tpm, efftx, ensg, ensgu, ntx_per_gene, counts, efftx)
    overdisp, prior = _moderate(overdisp, df)
    return _gene_output(counts, genelen, _rowsum(tpm, ensg), ensg, ensgu, codes, geneann,
                        maxtxlen.astype(np.int32), overdisp, prior, df, ["bootstrap"] * ns,
                        [_basename(p) for p in paths], divide, offset_prior)


def catchOarfishGene(parent_dir=None, prefixes=None, tx2gene=None, remove_version=True, divide=True,
                     gene_length="moderate", verbose=True):
    """Process Oarfish Output to Produce Gene Counts

    Produce a gene count object from Oarfish output for a series of biological samples, optionally with
    Gencode annotation.

    Parameters
    ----------
    parent_dir : str, optional
        Directory containing the Oarfish output files. Defaults to the current directory.
    prefixes : list of str, optional
        The sample-specific file prefixes created by Oarfish, one per RNA-seq sample; each may contain
        a directory separator. If ``parent_dir`` is given, they are relative to it. By default, the
        prefixes of all files in ``parent_dir`` with names ending in ``.quant`` are used, in sorted
        (C locale) order. For each prefix, ``<prefix>.quant``, ``<prefix>.meta_info.json`` and, when
        there are bootstrap samples, ``<prefix>.infreps.pq`` are read.
    tx2gene : DataFrame, optional
        Maps transcript IDs to gene IDs: the first column holds transcript IDs and the second the
        corresponding gene IDs. Not needed if Gencode annotation was used for the quantification.
    remove_version : bool, default True
        Whether to remove the version numbers from the transcript and gene IDs.
    divide : bool, default True
        Whether to divide the gene counts by the RTA overdispersions.
    gene_length : {"moderate", "tximport"}, default "moderate"
        How to compute the gene length (see Notes). ``"simple"`` raises ``NotImplementedError``: it
        fails in edgeR 4.99.16 ("length of 'dimnames' [2] not equal to array extent").
    verbose : bool, default True
        Whether to print progress information as each sample is read.

    Returns
    -------
    DGEList
        With gene IDs as row names, in order of first appearance, and components:

        ``counts``
            DataFrame of gene counts, one row per gene and one column per sample (named by the
            prefixes, without directories).
        ``samples``
            DataFrame with ``group``, ``lib.size`` (column sums of ``counts``) and ``norm.factors``.
        ``genes``
            DataFrame of gene annotation: ``GeneName`` (gene name), ``TxTypes`` (unique transcript
            biotypes for that gene, pipe-separated) and ``NPCTx`` (number of protein-coding
            transcripts) if Gencode annotation is used; then ``NTx`` (number of transcripts),
            ``MaxTxLen`` (maximum transcript length), ``AveEffLen`` (gene length, averaged over
            samples) and ``Overdispersion`` (genewise RTA overdispersion).
        ``overdispersion.prior``
            Median of the genewise overdispersions, used to moderate the overdispersion values (NaN
            when no sample has bootstrap samples).
        ``resample.type``
            List giving the resampling type (``"bootstrap"``) for each sample.
        ``divided.counts``
            Whether the counts were to be divided by the RTA overdispersions (the ``divide`` argument).
        ``other``
            Dict with ``"effective.length"``, the DataFrame of gene lengths, and ``"tpm"``, the
            DataFrame of genewise TPM values.

    Notes
    -----
    ``catchOarfishGene`` assumes that Oarfish (Jousheghani et al 2025) has been run to estimate
    transcript counts for one or more RNA samples and that bootstrap samples have also been generated.
    The function reads the Oarfish quantifications and summarizes them at the gene level. The
    bootstrap samples (Parquet files) are read with the optional package pyarrow
    (``pip install 'edgerpy[oarfish]'``); without it, ``ImportError`` is raised. No output files raise
    ``FileNotFoundError``, as does a missing ``.quant`` file.

    The transcript-level counts are collapsed to gene-level counts and RTA overdispersions are
    estimated at the gene level, using a method similar to Baldoni et al (2024ab) but applied to gene
    level counts instead of transcript counts. As in edgeR, whether the counts are divided depends on
    whether the last sample has bootstrap samples; if it has none, the counts are not divided and a
    message says so.

    The function needs to know which gene each transcript belongs to. If Oarfish was run with a
    Gencode transcript FASTA file, then ``catchOarfishGene`` detects that automatically. It then
    unpacks the Gencode transcript names using ``splitGencodeTxNames`` to obtain gene and transcript
    IDs. Otherwise, a DataFrame ``tx2gene`` needs to be provided to map transcript to gene IDs. If the
    transcript names are Gencode-style but those of ``tx2gene`` are not, the Ensembl transcript IDs
    are taken from the Gencode names. Without ``tx2gene``, names that are not Gencode-style raise
    ``ValueError``, as do transcripts missing from ``tx2gene`` and missing gene IDs.

    For long-read RNA-seq data, it is assumed that expected counts do not depend on the length of
    transcripts and that the effective length of transcripts equals their actual length. Therefore,
    ``catchOarfishGene`` does not create a prior offset matrix. Gene lengths are still computed using
    the method specified by ``gene_length``: the method of Soneson et al (2016) (``"tximport"``), or a
    method that moderates the lengths weakly towards the gene average and, even more weakly, towards
    the simple average of transcript lengths (``"moderate"``). As in edgeR 4.99.16, ``"tximport"``
    leaves the gene lengths equal to the gene counts when no gene has zero counts in all samples.

    The ``.quant`` files are parsed with readr's number parser, as R does, so every value is the one
    R reads.

    R documentation by Gordon Smyth and Pedro Baldoni; adapted for edgerpy.
    """
    _check_gene_length(gene_length)
    if gene_length == "simple":
        raise NotImplementedError("catchOarfishGene(gene_length='simple') fails in edgeR 4.99.16 "
                                  "(length of 'dimnames' [2] not equal to array extent)")
    prefixes = _oarfish_prefixes(parent_dir, prefixes)
    ns = len(prefixes)
    for j, pre in enumerate(prefixes):
        if verbose:
            print(f"Reading {pre}, ", end="")
        if not os.path.exists(f"{pre}.quant"):
            raise FileNotFoundError("quant file not found at specified path")
        nboot = _oarfish_nboot(pre)
        if verbose:
            print(f"{nboot} bootstraps")
        if j == 0:
            q = _read_tsv(f"{pre}.quant", "cdd")
            ntx = len(q["tname"])
            ensg, geneann = _tx2gene_map(q["tname"], tx2gene, remove_version, verbose)
            codes, ensgu = _group_codes(ensg)
            ng = len(ensgu)
            ntx_per_gene = _count_per_group(codes, ng)
            txcounts = np.zeros((ntx, ns), order="F")
            counts = np.zeros((ng, ns), order="F")
            df = np.zeros(ng, dtype=np.int64)
            overdisp = np.zeros(ng)
            txlen = q["len"]
            nr = q["num_reads"]
        else:
            nr = _read_tsv(f"{pre}.quant", "__d")["num_reads"]
        txcounts[:, j] = nr
        counts[:, j] = _rowsum(nr, ensg).reshape(-1)
        if nboot > 0:
            _accumulate(overdisp, df, np.asfortranarray(_rowsum(_read_parquet(f"{pre}.infreps.pq"), ensg)))
    maxtxlen = _max_by_group(txlen, codes, ng)
    tpm = (txcounts / _r.colSums(txcounts)[None, :]) * 1e6
    if gene_length == "tximport":
        genelen = _oarfish_tximport_length(txcounts, txlen, counts, ensg, ensgu, ntx_per_gene)
    else:
        genelen = _gene_length(gene_length, tpm, np.broadcast_to(txlen[:, None], tpm.shape), ensg, ensgu,
                               ntx_per_gene, counts, None)
    overdisp, prior = _moderate(overdisp, df)
    return _gene_output(counts, genelen, _rowsum(tpm, ensg), ensg, ensgu, codes, geneann, maxtxlen, overdisp, prior,
                        df, ["bootstrap"] * ns, [_basename(p) for p in prefixes], divide, False, range_col=False,
                        divide_ok=nboot > 0, divide_msg="No bootstrap samples, so counts not divided")


def _oarfish_tximport_length(txcounts, txlen, counts, ensg, ensgu, ntx_per_gene):
    """catchOarfishGene's gene.length="tximport", kept as edgeR 4.99.16 has it: R's x[-integer(0), ] selects
    nothing, so with no all-zero gene the lengths are left equal to the counts."""
    ensg_a = np.asarray(ensg, dtype=object)
    allzero = np.flatnonzero(_r.rowSums(counts) == 0)
    txz = np.flatnonzero(np.isin(ensg_a, np.asarray(ensgu, dtype=object)[allzero]))
    out = counts.copy()
    if allzero.size:
        out[allzero, :] = (_rowsum(txlen[txz], ensg_a[txz]).reshape(-1) / ntx_per_gene[allzero])[:, None]
        keep = np.setdiff1d(np.arange(len(ensg_a)), txz)
        rest = np.setdiff1d(np.arange(counts.shape[0]), allzero)
        with np.errstate(invalid="ignore", divide="ignore"):             # 0/0 is NaN, as in R; filled below
            out[rest, :] = (_rowsum(txcounts[keep] * txlen[keep][:, None], ensg_a[keep]) /
                            _rowsum(txcounts[keep], ensg_a[keep]))
    if np.isnan(out).any():
        m = _r.exp(_r.rowMeans(_r.log(out), na_rm=True))
        i = np.isnan(out)
        out[i] = np.broadcast_to(m[:, None], out.shape)[i]
    return out
