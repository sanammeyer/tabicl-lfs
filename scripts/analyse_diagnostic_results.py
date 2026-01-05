import argparse
import pandas as pd
import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

METRICS = ["acc", "f1", "nll", "ece"]

def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for m in METRICS:
        sa = f"{m}_sa"
        ea = f"{m}_ea"
        if sa in df.columns and ea in df.columns:
            df[f"delta_{m}"] = df[ea] - df[sa]
    return df

def summarize_wins_losses(ds_df: pd.DataFrame, key="delta_nll"):
    mean = ds_df[key].mean()
    n_better = int((ds_df[key] < 0).sum())
    n_total = int(ds_df.shape[0])
    return mean, n_better, n_total

def print_top(ds_df: pd.DataFrame, key="delta_nll", k=5, title=""):
    if title:
        print("\n" + title)
    print(f"Top {k} best (most negative {key}):")
    print(ds_df.sort_values(key).head(k))
    print(f"\nTop {k} worst (most positive {key}):")
    print(ds_df.sort_values(key, ascending=False).head(k))

def analyse_behaviour(path: str):
    beh = pd.read_csv(path)
    beh = add_deltas(beh)

    print("\n==================== BEHAVIOUR ====================")
    print("Rows:", len(beh))
    print("Datasets:", beh["dataset"].nunique())
    print("Context lengths (unique):", sorted(beh["context_length"].unique())[:20], " ...")

    # Per-dataset summary:
    # First average within each (dataset, seed, context_length, param_value) combination
    # to avoid overweighting datasets with more rows; then average across these units.
    group_cols = ["dataset"]
    for col in ["seed", "context_length", "param_value"]:
        if col in beh.columns:
            group_cols.append(col)

    beh_unit = beh.groupby(group_cols).agg(
        delta_nll=("delta_nll", "mean"),
        delta_ece=("delta_ece", "mean"),
        delta_acc=("delta_acc", "mean"),
        delta_f1=("delta_f1", "mean"),
    ).reset_index()

    ds = beh_unit.groupby("dataset").agg(
        n=("delta_nll", "size"),
        delta_nll=("delta_nll", "mean"),
        delta_ece=("delta_ece", "mean"),
        delta_acc=("delta_acc", "mean"),
        delta_f1=("delta_f1", "mean"),
    ).sort_values("delta_nll")

    mean, n_better, n_total = summarize_wins_losses(ds, "delta_nll")
    print(f"\nMean ΔNLL across datasets: {mean:+.6f}  (negative = EA better)")
    print(f"Datasets with ΔNLL < 0: {n_better}/{n_total}")
    print_top(ds, key="delta_nll", k=6, title="Per-dataset mean deltas (behaviour)")

    # Paired nonparametric test across datasets
    if wilcoxon is not None and len(ds) > 0:
        try:
            stat, p_value = wilcoxon(ds["delta_nll"])
            print(
                f"\nWilcoxon signed-rank test over datasets (ΔNLL): "
                f"statistic={stat:.3f}, p-value={p_value:.3g}"
            )
        except ValueError as exc:
            print(f"\nWilcoxon test for behaviour ΔNLL failed: {exc}")
    elif wilcoxon is None:
        print("\n[Info] SciPy not available; skipping Wilcoxon test for behaviour.")

    # Optional: if there is a real context-length sweep, this will show it
    ds_k = beh.groupby(["dataset", "context_length"]).agg(
        n=("delta_nll", "size"),
        delta_nll=("delta_nll", "mean"),
        delta_ece=("delta_ece", "mean"),
    ).reset_index()
    if beh["context_length"].nunique() > 1:
        multi = ds_k.groupby("dataset")["context_length"].nunique()
        n_multi = int((multi > 1).sum())
        print(f"\nDatasets with >1 context length: {n_multi}")
        print("Sample per-dataset/context-length deltas:")
        print(ds_k.sort_values(["dataset", "context_length"]).head(20))

    return beh, ds

def analyse_geometry(path: str):
    geo = pd.read_csv(path)

    print("\n==================== GEOMETRY ====================")
    print("Rows:", len(geo))
    print("Datasets:", geo["dataset"].nunique())

    # Mechanism deltas
    geo = geo.copy()
    geo["delta_neff_mean"] = geo["neff_ea_mean"] - geo["neff_sa_mean"]
    geo["delta_purity_top1"] = geo["purity_ea_top1"] - geo["purity_sa_top1"]
    geo["delta_purity_topk"] = geo["purity_ea_topk"] - geo["purity_sa_topk"]

    ds = geo.groupby("dataset").agg(
        delta_neff=("delta_neff_mean", "mean"),
        neff_sa=("neff_sa_mean", "mean"),
        neff_ea=("neff_ea_mean", "mean"),
        delta_purity1=("delta_purity_top1", "mean"),
        delta_purityk=("delta_purity_topk", "mean"),
        cka=("cka_sa_ea", "mean"),
        m_mean_ea=("m_mean_ea", "mean"),
        m_cv_ea=("m_cv_ea", "mean"),
    ).sort_values("delta_neff", ascending=False)

    print("\nPer-dataset mechanism summary (means over seeds):")
    print(ds)

    print("\nQuick stats:")
    print(f"Median ΔN_eff: {ds['delta_neff'].median():+.3f}")
    print(f"Mean CKA(SA,EA): {ds['cka'].mean():.3f}")
    print("\nLowest CKA datasets (most representational change):")
    print(ds.sort_values("cka").head(5)[["cka", "delta_neff", "delta_purity1", "delta_purityk"]])

    return geo, ds

def analyse_robustness(path: str):
    rob = pd.read_csv(path)
    rob = add_deltas(rob)

    print("\n==================== ROBUSTNESS ====================")
    print("Rows:", len(rob))
    print("Datasets:", rob["dataset"].nunique())
    print("Conditions:", rob["condition"].value_counts().to_dict())

    # Helper: dataset-level mean deltas for a selection
    def ds_mean(sub: pd.DataFrame, label: str):
        if sub.empty:
            print(f"\n--- {label} ---")
            print("No rows for this condition.")
            return pd.DataFrame()

        group_cols = ["dataset"]
        for col in ["seed", "context_length", "param_value"]:
            if col in sub.columns:
                group_cols.append(col)

        sub_unit = sub.groupby(group_cols).agg(
            delta_nll=("delta_nll", "mean"),
            delta_ece=("delta_ece", "mean"),
            delta_acc=("delta_acc", "mean"),
            delta_f1=("delta_f1", "mean"),
        ).reset_index()

        ds = sub_unit.groupby("dataset").agg(
            n=("delta_nll", "size"),
            delta_nll=("delta_nll", "mean"),
            delta_ece=("delta_ece", "mean"),
            delta_acc=("delta_acc", "mean"),
            delta_f1=("delta_f1", "mean"),
        ).sort_values("delta_nll")
        mean, n_better, n_total = summarize_wins_losses(ds, "delta_nll")
        print(f"\n--- {label} ---")
        print(f"Mean ΔNLL: {mean:+.6f} | ΔNLL<0 on {n_better}/{n_total} datasets")
        print_top(ds, key="delta_nll", k=6)
        return ds

    # Helper: robustness gain over clean (ΔΔNLL per dataset)
    def robustness_gain(ds_clean: pd.DataFrame, ds_corr: pd.DataFrame, label: str):
        if ds_clean is None or ds_corr is None or ds_clean.empty or ds_corr.empty:
            return None
        joined = ds_corr[["delta_nll"]].rename(columns={"delta_nll": "delta_nll_corr"}).join(
            ds_clean[["delta_nll"]].rename(columns={"delta_nll": "delta_nll_clean"}),
            how="inner",
        )
        if joined.empty:
            return None
        joined["ddelta_nll"] = joined["delta_nll_corr"] - joined["delta_nll_clean"]
        mean_gap, n_better_gap, n_total_gap = summarize_wins_losses(joined, "ddelta_nll")
        print(
            f"\nRobustness gain ΔΔNLL ({label} vs clean): "
            f"{mean_gap:+.6f} | ΔΔNLL<0 on {n_better_gap}/{n_total_gap} datasets"
        )
        return joined

    clean = rob[rob["condition"] == "clean"]
    ds_clean = ds_mean(clean, "clean")

    cp = rob[rob["condition"] == "context_poison"]
    # averaged over both fractions (0.05/0.10) and seeds
    ds_cp = ds_mean(cp, "context_poison (avg over param_value and seeds)")

    lp = rob[rob["condition"] == "label_poison"]
    ds_lp = ds_mean(lp, "label_poison")

    # Irrelevant features / uninformative feature injection
    # The CSV currently uses 'uninformative_features_shuffled' as the condition name.
    ir = rob[rob["condition"].isin(["irrelevant_features", "uninformative_features_shuffled"])]
    ds_ir = ds_mean(ir, "irrelevant_features (uninformative_features_shuffled)")

    # Robustness-specific effect: does EA help more under corruption than on clean?
    rg_cp = robustness_gain(ds_clean, ds_cp, "context_poison")
    rg_lp = robustness_gain(ds_clean, ds_lp, "label_poison")
    rg_ir = robustness_gain(ds_clean, ds_ir, "irrelevant_features (uninformative_features_shuffled)")

    # Wilcoxon tests on robustness gains (ΔΔNLL per dataset) where available
    if wilcoxon is not None:
        for rg, label in [
            (rg_cp, "context_poison"),
            (rg_lp, "label_poison"),
            (rg_ir, "irrelevant_features (uninformative_features_shuffled)"),
        ]:
            if rg is not None and "ddelta_nll" in rg.columns and len(rg) > 0:
                try:
                    stat, p_value = wilcoxon(rg["ddelta_nll"])
                    print(
                        f"Wilcoxon ΔΔNLL ({label} vs clean): "
                        f"statistic={stat:.3f}, p-value={p_value:.3g}"
                    )
                except ValueError as exc:
                    print(f"Wilcoxon test for {label} ΔΔNLL failed: {exc}")
    elif any(rg is not None for rg in [rg_cp, rg_lp, rg_ir]):
        print("\n[Info] SciPy not available; skipping Wilcoxon tests for robustness.")

    # Condition × severity summaries for main corruptions
    for sub, label in [
        (cp, "context_poison"),
        (lp, "label_poison"),
        (ir, "irrelevant_features (uninformative_features_shuffled)"),
    ]:
        if not sub.empty and "param_value" in sub.columns:
            print(f"\n--- {label}: mean ΔNLL by param_value ---")
            summary = sub.groupby(["condition", "param_value"]).agg(
                delta_nll=("delta_nll", "mean"),
                delta_ece=("delta_ece", "mean"),
                n=("delta_nll", "size"),
                n_datasets=("dataset", "nunique"),
            ).reset_index().sort_values(["condition", "param_value"])
            print(summary)

    # Rotations: show summary by (condition, k/angle)
    rot = rob[rob["condition"].str.contains("feature_rotation", na=False)]
    if len(rot) > 0:
        print("\n--- feature rotations: mean ΔNLL by condition and param_value ---")
        rot_summary = rot.groupby(["condition", "param_value"]).agg(
            delta_nll=("delta_nll", "mean"),
            delta_ece=("delta_ece", "mean"),
            n=("delta_nll", "size"),
            n_datasets=("dataset", "nunique"),
        ).reset_index().sort_values(["condition", "param_value"])
        print(rot_summary)

        # If k=8 exists, print per-dataset for k=8 (usually the main comparable case)
        k8_mask = np.isclose(rot["param_value"].astype(float), 8.0)
        if k8_mask.any():
            rot_k8 = rot[k8_mask]
            print("\nPer-dataset ΔNLL at k=8 (mean over seeds/rot draws):")
            rot_k8_ds = rot_k8.groupby(["condition", "dataset"]).agg(
                delta_nll=("delta_nll", "mean"),
                n=("delta_nll", "size"),
            ).reset_index().sort_values(["condition", "delta_nll"])
            print(rot_k8_ds)

    return rob, (ds_clean, ds_cp, ds_lp, ds_ir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--robustness", required=True)
    args = ap.parse_args()

    beh, ds_beh = analyse_behaviour(args.behaviour)
    geo, ds_geo = analyse_geometry(args.geometry)

    # Correlate per-dataset ΔN_eff (geometry) with ΔNLL (behaviour)
    try:
        merged = ds_geo[["delta_neff"]].join(ds_beh[["delta_nll"]], how="inner")
        if len(merged) > 1:
            corr = merged["delta_neff"].corr(merged["delta_nll"])
            print(
                "\nCorrelation between per-dataset ΔN_eff and ΔNLL (behaviour): "
                f"{corr:+.3f}  "
                "(positive = larger ΔN_eff associated with worse ΔNLL)"
            )
        else:
            print("\n[Info] Not enough datasets for ΔN_eff vs ΔNLL correlation.")
    except Exception as exc:
        print(f"\n[Info] Failed to compute ΔN_eff vs ΔNLL correlation: {exc}")

    analyse_robustness(args.robustness)

if __name__ == "__main__":
    main()
