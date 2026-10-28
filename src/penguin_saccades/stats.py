# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — stats                                        ║
# ║  « run everything, decide later »                                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Bootstrap median confidence intervals, summary tables with n,   ║
# ║  and pairwise Fisher re-randomisation tests with Benjamini-      ║
# ║  Hochberg correction via reRandomStats. Seeds are fixed so       ║
# ║  every p-value in the manuscript can be regenerated.             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Summary statistics and re-randomisation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from penguin_saccades import constants as C


# ┌────────────────────────────────────────────────────────────┐
# │ Bootstrap  « median and its confidence interval »          │
# └────────────────────────────────────────────────────────────┘
def bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int | None = None,
    confidence: float = C.CONFIDENCE_LEVEL,
    seed: int = C.RANDOM_SEED,
) -> tuple[float, float, float]:
    """Return ``(median, lower, upper)`` from a percentile bootstrap; NaN-safe.

    ``n_boot`` defaults to ``constants.N_BOOTSTRAP`` at call time.
    """
    n_boot = C.N_BOOTSTRAP if n_boot is None else n_boot
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    medians = np.median(x[idx], axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(medians, [tail, 1.0 - tail])
    return float(np.median(x)), float(lo), float(hi)


def summarise(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    """Per group: n, median with bootstrap CI, mean, SEM, min, max."""
    rows: list[dict] = []
    for key, grp in df.groupby(group_cols, sort=True, dropna=False):
        x = grp[value_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        med, lo, hi = bootstrap_median_ci(x)
        rows.append({
            **dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
            "variable": value_col,
            "n": int(x.size),
            "median": med,
            "ci_lower": lo,
            "ci_upper": hi,
            "mean": float(np.mean(x)) if x.size else np.nan,
            "sem": float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan,
            "min": float(np.min(x)) if x.size else np.nan,
            "max": float(np.max(x)) if x.size else np.nan,
        })
    return pd.DataFrame(rows)


# ┌────────────────────────────────────────────────────────────────┐
# │ Re-randomisation  « pairwise Fisher tests with BH correction » │
# └────────────────────────────────────────────────────────────────┘
def pairwise_fisher(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    statistic: str = "medianDiff",
    n_permutations: int | None = None,
    seed: int = C.RANDOM_SEED,
) -> pd.DataFrame:
    """All pairwise group comparisons with Fisher resampling and Benjamini-Hochberg FDR.

    Groups with fewer than two finite values are dropped before testing.
    ``n_permutations`` defaults to ``constants.N_PERMUTATIONS`` at call time.
    """
    from rerandomstats import MultiGroupTest

    n_permutations = C.N_PERMUTATIONS if n_permutations is None else n_permutations

    clean = df[[group_col, value_col]].dropna()
    counts = clean.groupby(group_col)[value_col].count()
    clean = clean[clean[group_col].isin(counts[counts >= 2].index)]
    if clean[group_col].nunique() < 2:
        return pd.DataFrame()
    test = MultiGroupTest(
        clean[value_col].to_list(),
        clean[group_col].astype(str).to_list(),
        f"Fisher:{statistic}",
        combination_n=n_permutations,
        correction_type="fdr_bh",
        **_seed_kwarg(MultiGroupTest, seed),
    )
    result = test.main()
    result["variable"] = value_col
    return result


def two_sample_fisher(
    a: np.ndarray,
    b: np.ndarray,
    statistic: str = "medianDiff",
    n_permutations: int | None = None,
    seed: int = C.RANDOM_SEED,
) -> float:
    """Two-sided p-value of a Fisher resampling test between two samples."""
    from rerandomstats import FisherResamplingTest

    n_permutations = C.N_PERMUTATIONS if n_permutations is None else n_permutations

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    test = FisherResamplingTest(
        a[np.isfinite(a)].tolist(), b[np.isfinite(b)].tolist(), statistic, n_permutations,
        **_seed_kwarg(FisherResamplingTest, seed),
    )
    return float(test.main())


def _seed_kwarg(cls: type, seed: int) -> dict[str, int]:
    """``{"seed": seed}`` when the installed reRandomStats accepts it, else a loud warning."""
    import inspect
    import warnings

    if "seed" in inspect.signature(cls.__init__).parameters:
        return {"seed": seed}
    warnings.warn(
        "Installed reRandomStats has no 'seed' argument; p-values will not be exactly "
        "reproducible.  Upgrade: pip install -U git+https://github.com/zerotonin/reRandomStats.git",
        RuntimeWarning, stacklevel=3,
    )
    return {}
