"""edgerpy: the edgeR v4 quasi-likelihood pipelines for Python.

The numerical work runs in edgeR's own C code (with limma's, R's nmath and
LINPACK, and Netlib BLAS/LAPACK), compiled from verbatim sources. Function
names, defaults and result components follow R; argument names use '_' for
R's '.', and integer coef indices are 0-based.

NB QL:        DGEList -> filterByExpr -> normLibSizes -> glmQLFit -> glmQLFTest -> topTags
Binomial QL:  PCList -> binQLFit -> binQLFTest -> topTags
diffSplice:   DGEList -> ... -> glmQLFit -> diffSplice -> topSplice

Modules: one per pipeline step (DGEList, filterByExpr, normLibSizes, aveLogCPM,
glmfit, glmQLFTest, binQLFtest, diffSplice, topTags), classes for the R
objects, helpers and _squeezeVar for the shared building blocks (not exported),
and _r for R's numerical primitives.
"""

from . import _C
from .aveLogCPM import aveLogCPM, cpm
from .binQLFtest import binFit, binQLFit, binQLFTest
from .catch import (catchKallisto, catchKallistoGene, catchOarfish, catchOarfishGene, catchRSEM, catchSalmon,
                    catchSalmonGene, splitGencodeTxNames)
from .classes import DGEBIN, DGEDTU, DGEGLM, DGELRT, MDS, TestResults, TopTags
from .classes import DGEList as DGEListClass
from .classes import PCList as PCListClass
from .DGEList import DGEList, PCList, getOffset
from .decideTests import decideTests
from .diffSplice import diffSplice, topSplice
from .estimateDisp import estimateDisp
from .filterByExpr import filterByExpr
from .glmfit import glmFit
from .glmQLFTest import glmQLFit, glmQLFTest
from .helpers import CompressedMatrix, modelMatrix
from .normLibSizes import normLibSizes
from .plotBCV import plotBCV
from .plotMD import plotMD
from .plotMDS import plotMDS
from .plotQLDisp import plotQLDisp
from .plotSplice import plotSplice
from .sampleWeights import sampleWeights
from .topTags import topTags

__version__ = "0.1.0"

__all__ = [
    # edgeR pipeline functions
    "DGEList", "PCList", "filterByExpr", "normLibSizes", "aveLogCPM", "cpm", "getOffset",
    "glmFit", "glmQLFit", "glmQLFTest", "sampleWeights", "binFit", "binQLFit", "binQLFTest", "diffSplice", "topTags",
    "decideTests",
    "catchSalmon", "catchKallisto", "catchRSEM", "catchOarfish", "catchSalmonGene", "catchKallistoGene",
    "catchOarfishGene", "splitGencodeTxNames",
    "plotMDS", "plotMD", "plotQLDisp", "estimateDisp", "plotBCV",
    "CompressedMatrix",
    # limma functions
    "topSplice", "plotSplice",
    # R stats
    "modelMatrix",
    # R object classes
    "DGEListClass", "PCListClass", "DGEGLM", "DGEBIN", "DGELRT", "TopTags", "DGEDTU", "MDS", "TestResults",
]
