"""Statistical agreement analysis: Pearson, ICC, Lin's CCC, Bland-Altman, paired t-test."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
import pingouin as pg


def lins_ccc(y1, y2):
    """Lin's concordance correlation coefficient."""
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    mean1, mean2 = np.mean(y1), np.mean(y2)
    var1, var2 = np.var(y1, ddof=1), np.var(y2, ddof=1)
    cov = np.cov(y1, y2)[0, 1]
    ccc = (2 * cov) / (var1 + var2 + (mean1 - mean2) ** 2)
    return ccc


def bland_altman(model_values, expert_values):
    """Compute Bland-Altman statistics."""
    model_values = np.asarray(model_values, dtype=float)
    expert_values = np.asarray(expert_values, dtype=float)
    diff = model_values - expert_values
    mean_both = (model_values + expert_values) / 2
    bias = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff
    return {
        "bias": round(float(bias), 4),
        "sd_diff": round(float(sd_diff), 4),
        "loa_lower": round(float(loa_lower), 4),
        "loa_upper": round(float(loa_upper), 4),
        "mean_both": mean_both.tolist(),
        "diff": diff.tolist(),
    }


def compute_icc(model_values, expert_values, image_ids):
    """Compute ICC (two-way mixed, absolute agreement, single measures)."""
    df = pd.DataFrame({
        "image_id": list(image_ids) + list(image_ids),
        "source": ["model"] * len(image_ids) + ["expert"] * len(image_ids),
        "value": list(model_values) + list(expert_values),
    })
    try:
        icc_result = pg.intraclass_corr(data=df, targets="image_id", raters="source", ratings="value")
        # Use ICC(A,1) — two-way mixed, absolute agreement, single measures
        icc_row = icc_result[icc_result["Type"] == "ICC(A,1)"]
        if len(icc_row) == 0:
            # Fallback to ICC(C,1)
            icc_row = icc_result[icc_result["Type"] == "ICC(C,1)"]
        if len(icc_row) > 0:
            ci_col = "CI95" if "CI95" in icc_result.columns else "CI95%"
            ci = icc_row[ci_col].values[0]
            return {
                "icc": round(float(icc_row["ICC"].values[0]), 4),
                "icc_ci_lower": round(float(ci[0]), 4),
                "icc_ci_upper": round(float(ci[1]), 4),
            }
    except Exception as e:
        print(f"  ICC computation failed: {e}")
    return {"icc": np.nan, "icc_ci_lower": np.nan, "icc_ci_upper": np.nan}


def compute_agreement(model_values, expert_values, image_ids, placenta_ids=None):
    """Compute all agreement statistics for one metric.

    Returns dict with all agreement metrics.
    """
    model_values = np.asarray(model_values, dtype=float)
    expert_values = np.asarray(expert_values, dtype=float)

    # Pearson correlation
    r, p = pearsonr(model_values, expert_values)

    # ICC
    icc_result = compute_icc(model_values, expert_values, image_ids)

    # Lin's CCC
    ccc = lins_ccc(model_values, expert_values)

    # Bland-Altman
    ba = bland_altman(model_values, expert_values)

    # Mean percent difference
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = np.where(expert_values != 0,
                            (model_values - expert_values) / expert_values * 100, 0)
    mean_pct_diff = np.mean(pct_diff)

    result = {
        "pearson_r": round(float(r), 4),
        "pearson_p": float(p),
        "icc": icc_result["icc"],
        "icc_ci_lower": icc_result["icc_ci_lower"],
        "icc_ci_upper": icc_result["icc_ci_upper"],
        "lins_ccc": round(float(ccc), 4),
        "bias": ba["bias"],
        "loa_lower": ba["loa_lower"],
        "loa_upper": ba["loa_upper"],
        "mean_pct_diff": round(float(mean_pct_diff), 4),
    }

    # Paired t-test at placenta level (n=5)
    if placenta_ids is not None:
        df = pd.DataFrame({
            "placenta_id": placenta_ids,
            "model": model_values,
            "expert": expert_values,
        })
        placenta_model = df.groupby("placenta_id")["model"].mean()
        placenta_expert = df.groupby("placenta_id")["expert"].mean()
        t_stat, t_p = ttest_rel(placenta_model, placenta_expert)
        result["paired_t_stat"] = round(float(t_stat), 4)
        result["paired_t_p"] = float(t_p)
    else:
        result["paired_t_stat"] = np.nan
        result["paired_t_p"] = np.nan

    return result
