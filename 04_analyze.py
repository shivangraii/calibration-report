"""Step 4: ECE, Brier, AUROC, bin table, reliability diagram, subgroup breakdowns."""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import config


# ---------------------------------------------------------------- metrics
def bin_table(conf, y, n_bins):
    """Equal-width bins (lo, hi]; a confidence of exactly 0 falls in the first bin."""
    idx = np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    rows = []
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        rows.append({"bin": f"{k * 100 // n_bins}-{(k + 1) * 100 // n_bins}%", "n": n,
                     "mean_conf": conf[m].mean() if n else np.nan,
                     "accuracy": y[m].mean() if n else np.nan})
    t = pd.DataFrame(rows)
    t["gap"] = t["mean_conf"] - t["accuracy"]      # > 0 means overconfident
    return t


def ece_from_table(t):
    s = t.dropna()
    return float((s["n"] / s["n"].sum() * (s["accuracy"] - s["mean_conf"]).abs()).sum())


def ece(conf, y, n_bins):
    return ece_from_table(bin_table(conf, y, n_bins))


def ece_equal_mass(conf, y, n_bins):
    chunks = np.array_split(np.argsort(conf), n_bins)
    return float(sum(len(c) / len(conf) * abs(y[c].mean() - conf[c].mean()) for c in chunks if len(c)))


def brier(conf, y):
    return float(np.mean((conf - y) ** 2))


def auroc(conf, y):
    return float(roc_auc_score(y, conf)) if 0 < y.sum() < len(y) else float("nan")


def bootstrap_ci(conf, y, n_bins, B, seed=config.SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        vals.append(ece(conf[i], y[i], n_bins))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def summarize(conf, y, n_bins):
    return {"n": int(len(y)), "accuracy": float(y.mean()), "mean_conf": float(conf.mean()),
            "overconfidence_gap": float(conf.mean() - y.mean()),
            "ece": ece(conf, y, n_bins), "ece_equal_mass": ece_equal_mass(conf, y, n_bins),
            "brier": brier(conf, y), "auroc": auroc(conf, y)}


def group_table(df, col, conf_col, n_bins):
    rows = []
    for g, s in df.groupby(col):
        c, y = s[conf_col].to_numpy(), s["y"].to_numpy()
        r = {"group": g, "n": len(s), "accuracy": y.mean(), "mean_conf": c.mean(),
             "gap": c.mean() - y.mean(), "brier": brier(c, y), "auroc": auroc(c, y),
             "ece": ece(c, y, n_bins) if len(s) >= config.MIN_GROUP_N else np.nan}
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------- plotting
def plot_reliability(t, conf, title, stem, n_bins):
    s = t.dropna()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(s["mean_conf"], s["accuracy"], "-", color="tab:blue", alpha=0.6)
    ax.scatter(s["mean_conf"], s["accuracy"], s=20 + 400 * s["n"] / s["n"].sum(),
               color="tab:blue", zorder=3, label="Bins (size ~ count)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax2.hist(conf, bins=np.linspace(0, 1, n_bins + 1), color="tab:gray", edgecolor="white")
    ax2.set_xlabel("Mean confidence"); ax2.set_ylabel("Count")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.ANSWER_MODEL)
    ap.add_argument("--mode", choices=["logprob", "verbal"], default="logprob")
    ap.add_argument("--conf", default=None, help="confidence column (default: conf_mean / conf_verbal)")
    args = ap.parse_args()

    p = config.paths(args.model, args.mode)
    conf_col = args.conf or ("conf_verbal" if args.mode == "verbal" else "conf_mean")
    out = p["analysis"]; out.mkdir(parents=True, exist_ok=True)

    qs = pd.read_csv(config.PROCESSED_CSV)
    ans = pd.read_json(p["answers"], lines=True)
    gr = pd.read_json(p["graded"], lines=True)[["id", "grade", "grade_method"]]
    df = qs.merge(ans, on="id").merge(gr, on="id")
    df = df[df["grade"].isin(["CORRECT", "INCORRECT", "NOT_ATTEMPTED"])].copy()
    df["y"] = (df["grade"] == "CORRECT").astype(int)      # NOT_ATTEMPTED counts as 0
    n_before = len(df)
    df = df.dropna(subset=[conf_col]).copy()
    print(f"{len(df)} usable rows ({n_before - len(df)} dropped for missing confidence)")

    conf, y, K = df[conf_col].to_numpy(float), df["y"].to_numpy(int), config.N_BINS
    t = bin_table(conf, y, K)
    m = summarize(conf, y, K)
    m["ece_95ci"] = bootstrap_ci(conf, y, K, config.N_BOOTSTRAP)
    m["conf_column"] = conf_col
    m["answer_model"] = args.model
    m["grade_counts"] = {k: int(v) for k, v in df["grade"].value_counts().items()}

    # variants of the confidence definition
    variants = [c for c in ["conf_mean", "conf_geo", "conf_first", "conf_min", "conf_verbal"] if c in df.columns and df[c].notna().all()]
    var_rows = [{"confidence": c, **summarize(df[c].to_numpy(float), y, K)} for c in variants]

    # agreement with the manual audit, if filled in
    if p["audit"].exists():
        a = pd.read_csv(p["audit"]).dropna(subset=["human_grade"])
        a = a[a["human_grade"].astype(str).str.strip() != ""]
        if len(a):
            m["audit_n"] = int(len(a))
            m["audit_agreement"] = float((a["auto_grade"].str.upper() == a["human_grade"].str.upper().str.strip()).mean())

    # save everything
    t.to_csv(out / "bin_table.csv", index=False)
    pd.DataFrame(var_rows).to_csv(out / "confidence_variants.csv", index=False)
    group_table(df, "difficulty_merged", conf_col, K).to_csv(out / "by_difficulty.csv", index=False)
    group_table(df, "category_merged", conf_col, K).to_csv(out / "by_category.csv", index=False)
    df.to_csv(out / "results_merged.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    plot_reliability(t, conf, f"Reliability diagram ({args.model}, {conf_col})", str(out / "reliability"), K)

    print(json.dumps(m, indent=2))
    print("\n", t.round(3).to_string(index=False))
    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
