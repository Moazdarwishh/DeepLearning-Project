"""Full exploratory data analysis (Milestone 1, Task 1.1).

``python -m src.cli data --action eda`` runs everything and writes:

    outputs/eda/image_stats.csv          one row per image (train + test)
    outputs/eda/eda_summary.json         every number used in the report
    outputs/eda/spell_fixes.json         typo -> correction map used for cleaning
    outputs/eda/figures/*.png            all EDA figures
    data/processed/captions_clean.json   {"train": {id: [10 captions]}, "test": {...}}
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .captions import build_spell_fixes, caption_statistics, clean_all
from .datasets import get_class_names, load_cub
from .image_stats import find_near_duplicates, scan_images
from .splits import load_split


def _quality_thresholds(df) -> dict:
    return {
        "dark": 0.15,
        "bright": 0.85,
        "low_contrast": 0.06,
        "blurry": float(np.percentile(df["blur_laplacian_var"].dropna(), 1)),  # bottom 1 %
        "small_side": 150,
        "gray_colorful_fraction": 0.001,   # < 0.1 % of pixels carry any colour -> grayscale
        "duplicate_exact": 2,         # pHash Hamming distance for a confident duplicate
        "duplicate_candidate": 6,     # looser candidates, reported for manual review
    }


def _flag_quality(df, t: dict):
    df = df.copy()
    df["is_grayscale_mode"] = df["mode"] == "L"
    df["is_grayscale_content"] = (df["mode"] != "L") & (df["colorful_fraction"] < t["gray_colorful_fraction"])
    df["is_dark"] = df["brightness"] < t["dark"]
    df["is_bright"] = df["brightness"] > t["bright"]
    df["is_low_contrast"] = df["contrast"] < t["low_contrast"]
    df["is_blurry"] = df["blur_laplacian_var"] < t["blurry"]
    df["is_small"] = df[["width", "height"]].min(axis=1) < t["small_side"]
    return df


def _describe(series) -> dict:
    s = series.dropna()
    return {"min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
            "mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3)}


def run_eda(cfg, max_grid: int = 12) -> dict:
    import pandas as pd

    from src.visualization import eda_plots as P

    out = Path(cfg.experiment.output_dir) / "eda"
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.experiment.seed)

    train = load_cub(cfg.data.train_dataset, cfg.data.cache_dir)
    test = load_cub(cfg.data.test_dataset, cfg.data.cache_dir)
    split = load_split(cfg.data.splits_dir, expected_rows=len(train))
    names = get_class_names(train)
    dsets = {"train": train, "test": test}

    # ------------------------------------------------------------------ images
    df_train, channel = scan_images(train, "train", set(split.train_indices))
    df_test, _ = scan_images(test, "test")
    df_train.loc[df_train["index"].isin(split.val_indices), "split"] = "val"
    df = pd.concat([df_train, df_test], ignore_index=True)

    corrupt = df[df["corrupt"]]
    ok = df[~df["corrupt"]].copy()
    thr = _quality_thresholds(ok)
    ok = _flag_quality(ok, thr)
    ok.to_csv(out / "image_stats.csv", index=False)

    def source(split_name):  # val rows live in the HF train set
        return "test" if split_name == "test" else "train"

    def img(row):
        return dsets[source(row["split"])][int(row["index"])]["image"]

    def title(row, extra=""):
        return f"{names[int(row['label'])][:22]}\n{row['split']}#{int(row['index'])} {extra}".strip()

    # class distribution
    counts = {s: np.bincount(ok.loc[ok["split"] == s, "label"].astype(int), minlength=200)
              for s in ("train", "val", "test")}
    figures = [P.class_distribution(counts, fig_dir), P.image_sizes(ok, fig_dir),
               P.quality(ok, fig_dir, thr)]

    # grayscale images
    gray = ok[ok["is_grayscale_mode"] | ok["is_grayscale_content"]]
    sample = gray.sample(min(max_grid, len(gray)), random_state=0) if len(gray) else gray
    figures.append(P.image_grid([img(r) for _, r in sample.iterrows()],
                                [title(r, r["mode"]) for _, r in sample.iterrows()],
                                fig_dir, "04_grayscale_images.png",
                                f"Grayscale images ({len(gray)} total)"))

    # lowest-quality examples
    worst = pd.concat([ok.nsmallest(4, "blur_laplacian_var").assign(why="blurry"),
                       ok.nsmallest(4, "brightness").assign(why="dark"),
                       ok.nsmallest(4, "contrast").assign(why="low contrast")])
    figures.append(P.image_grid([img(r) for _, r in worst.iterrows()],
                                [title(r, r["why"]) for _, r in worst.iterrows()],
                                fig_dir, "05_low_quality_examples.png",
                                "Lowest-quality images (blur / darkness / contrast)"))

    # duplicates (perceptual hash, Hamming distance <= 4)
    tr_rows = ok[ok["split"].isin(["train", "val"])].reset_index(drop=True)
    te_rows = ok[ok["split"] == "test"].reset_index(drop=True)
    cand = thr["duplicate_candidate"]
    dup_rows = []
    for kind, a_df, b_df, found in (
        ("train/val-train/val", tr_rows, tr_rows, find_near_duplicates(tr_rows["phash"].tolist(), max_distance=cand)),
        ("test-test", te_rows, te_rows, find_near_duplicates(te_rows["phash"].tolist(), max_distance=cand)),
        ("train/val-TEST", tr_rows, te_rows,
         find_near_duplicates(tr_rows["phash"].tolist(), te_rows["phash"].tolist(), max_distance=cand)),
    ):
        for i, j, d in found:
            a, b = a_df.iloc[i], b_df.iloc[j]
            dup_rows.append({"kind": kind, "distance": d, "confident": d <= thr["duplicate_exact"],
                             "a_split": a["split"], "a_index": int(a["index"]), "a_label": names[int(a["label"])],
                             "b_split": b["split"], "b_index": int(b["index"]), "b_label": names[int(b["label"])],
                             "label_conflict": int(a["label"]) != int(b["label"])})
    dups = pd.DataFrame(dup_rows, columns=["kind", "distance", "confident", "a_split", "a_index", "a_label",
                                           "b_split", "b_index", "b_label", "label_conflict"])
    dups.sort_values(["confident", "label_conflict", "distance"], ascending=[False, False, True]) \
        .to_csv(out / "near_duplicates.csv", index=False)
    confident = dups[dups["confident"]]

    def _row(split_name, index):
        return ok[(ok["split"] == split_name) & (ok["index"] == index)].iloc[0]

    pairs = []
    for _, r in confident.sort_values(["label_conflict", "kind"], ascending=[False, True]).head(10).iterrows():
        a, b = _row(r["a_split"], r["a_index"]), _row(r["b_split"], r["b_index"])
        tag = "LABEL CONFLICT" if r["label_conflict"] else r["kind"]
        pairs.append((img(a), title(a, tag), img(b), title(b, f"dist={r['distance']}")))
    figures.append(P.pair_grid(pairs, fig_dir, "06_near_duplicates.png",
                               f"Confident duplicates (pHash distance <= {thr['duplicate_exact']})"))
    rejected = dups[~dups["confident"]].head(4)
    pairs = []
    for _, r in rejected.iterrows():
        a, b = _row(r["a_split"], r["a_index"]), _row(r["b_split"], r["b_index"])
        pairs.append((img(a), title(a, r["kind"]), img(b), title(b, f"dist={r['distance']}")))
    figures.append(P.pair_grid(pairs, fig_dir, "06b_duplicate_false_positives.png",
                               "Looser matches: mostly similar plain backgrounds, not duplicates"))

    # sample grid + fine-grained example (all gulls)
    picks = rng.sample(range(200), 18)
    rows = [ok[(ok["split"] == "train") & (ok["label"] == c)].iloc[0] for c in picks]
    figures.append(P.image_grid([img(r) for r in rows], [names[int(r["label"])] for r in rows],
                                fig_dir, "07a_sample_grid.png", "Random classes (one image each)"))
    gulls = [i for i, n in enumerate(names) if "Gull" in n]
    rows = [ok[(ok["split"] == "train") & (ok["label"] == c)].iloc[k] for c in gulls for k in (0, 1)]
    figures.append(P.image_grid([img(r) for r in rows], [names[int(r["label"])] for r in rows],
                                fig_dir, "07b_fine_grained_gulls.png",
                                f"Fine-grained difficulty: the {len(gulls)} gull species look alike",
                                ncols=4))

    # ---------------------------------------------------------------- captions
    cap_stats = {s: caption_statistics(d["description"]) for s, d in dsets.items()}
    lengths_all = cap_stats["train"]["_lengths"] + cap_stats["test"]["_lengths"]
    p95 = float(np.percentile(lengths_all, 95))
    figures.append(P.caption_lengths({s: v["_lengths"] for s, v in cap_stats.items()}, p95, fig_dir))
    figures.append(P.top_words(cap_stats["train"]["top_words"], fig_dir))
    color_counts = Counter(cap_stats["train"]["color_word_counts"]) + Counter(cap_stats["test"]["color_word_counts"])
    figures.append(P.color_words(dict(color_counts), fig_dir))

    # spell fixes are learned from the TRAIN captions only, then applied to both splits
    fixes = build_spell_fixes(cap_stats["train"]["_word_counts"], cap_stats["train"]["_bigram_counts"])
    fix_kinds = {"merged_words_split": sum(" " in v for v in fixes.values()),
                 "typos_corrected": sum(" " not in v for v in fixes.values())}
    (out / "spell_fixes.json").write_text(json.dumps(fixes, indent=1, sort_keys=True))
    clean = {s: clean_all(d["id"], d["description"], fixes) for s, d in dsets.items()}
    clean_path = Path("data/processed/captions_clean.json")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(json.dumps(clean))
    clean_vocab = Counter(w for caps in clean["train"].values() for c in caps for w in c.split())

    for s in cap_stats.values():
        s.pop("_lengths"); s.pop("_word_counts"); s.pop("_bigram_counts")

    # ----------------------------------------------------------------- summary
    flags = [c for c in ok.columns if c.startswith("is_")]
    summary = {
        "rows": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "corrupt_images": corrupt.reindex(columns=["split", "index", "error"]).to_dict("records"),
        "images_per_class": {s: {"min": int(c.min()), "max": int(c.max()), "mean": round(float(c.mean()), 2),
                                 "classes_below_20": int((c < 20).sum())} for s, c in counts.items()},
        "smallest_test_classes": [{"class": names[i], "count": int(counts["test"][i])}
                                  for i in np.argsort(counts["test"])[:5]],
        "image_size": {"width": _describe(ok["width"]), "height": _describe(ok["height"]),
                       "aspect": _describe(ok["aspect"])},
        "color_modes": {s: sub["mode"].value_counts().to_dict() for s, sub in ok.groupby("split")},
        "quality_thresholds": thr,
        "quality_flags": {s: {f: int(sub[f].sum()) for f in flags} for s, sub in ok.groupby("split")},
        "quality_stats": {c: _describe(ok[c]) for c in ("brightness", "contrast", "blur_laplacian_var")},
        "near_duplicates": {
            "confident_by_kind": confident["kind"].value_counts().to_dict(),
            "candidates_by_kind": dups["kind"].value_counts().to_dict(),
            "label_conflicts": confident[confident["label_conflict"]].to_dict("records"),
            "leaks_train_val_to_test": confident[confident["kind"] == "train/val-TEST"].to_dict("records"),
            "csv": str(out / "near_duplicates.csv"),
        },
        "train_channel_mean_std": channel,
        "captions": cap_stats,
        "caption_length_p95_all": p95,
        "cleaning": {"spell_fixes": len(fixes), "spell_fix_kinds": fix_kinds, "spell_fix_examples": dict(list(fixes.items())[:25]),
                     "vocab_raw_train": cap_stats["train"]["vocab_size"], "vocab_clean_train": len(clean_vocab),
                     "output": str(clean_path)},
        "figures": [str(f) for f in figures if f],
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
