#!/usr/bin/env python3
"""
summarise_results.py
────────────────────
Loads experiment results from Google Drive and prints a formatted
summary table to the terminal. Useful for a quick check after the
notebook finishes, or for partial runs.

Usage (Colab terminal or local after Drive sync):
    python scripts/summarise_results.py
    python scripts/summarise_results.py --results_dir /path/to/results
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v:.2%}"
    return f"{v:.4f}"


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
                      for i in range(len(headers))]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    row_fmt = "|" + "|".join(f" {{:<{w-1}}}" for w in col_widths) + "|"
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)
    for row in rows:
        print(row_fmt.format(*[str(x) for x in row]))
    print(sep)


def main(results_dir: Path):
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Make sure the notebook has run and Drive is mounted.")
        return

    found = list(results_dir.glob("*.json"))
    if not found:
        print(f"No JSON files found in {results_dir}")
        return

    print(f"\nLoading results from: {results_dir}")
    print(f"Files found: {len(found)}\n")

    # ── Main benchmark ────────────────────────────────────────────────────────
    main_keys = [
        ("DistilBERT_prompt",   "DistilBERT+P"),
        ("DistilRoBERTa_prompt","DiRoBERTa+P"),
        ("RoBERTa-base_prompt", "RoBERTa+P"),
        ("ALBERT-base-v2_prompt","ALBERT+P"),
        ("ELECTRA-base_prompt", "ELECTRA+P"),
        ("DistilBERT_noprompt", "DistilBERT-NP"),
        ("DistilRoBERTa_noprompt","DiRoBERTa-NP"),
        ("ELECTRA-base_noprompt","ELECTRA-NP"),
    ]

    rows = []
    for key, label in main_keys:
        path = results_dir / f"{key}_full.json"
        if path.exists():
            m = load_json(path)
            tput = m.get("throughput_sps", m.get("throughput", "—"))
            tput_str = f"{int(tput):,}" if isinstance(tput, (int, float)) else str(tput)
            rows.append([label, fmt(m["accuracy"]), fmt(m["macro_f1"]),
                         fmt(m["weighted_f1"]), tput_str])
        else:
            rows.append([label, "—", "—", "—", "—"])

    print("── Main Benchmark (seed 42, Template A) ──")
    print_table(
        ["Model", "Acc", "Mac-F1", "W-F1", "Throughput"],
        rows,
        col_widths=[18, 8, 8, 8, 14],
    )

    # ── Multi-seed ────────────────────────────────────────────────────────────
    ms_path = results_dir / "multiseed_aggregated.json"
    if ms_path.exists():
        ms = load_json(ms_path)
        print("\n── Multi-Seed Robustness (seeds 42, 2024, 7) ──")
        ms_rows = []
        for model, stats in ms.items():
            acc_mean = stats.get("accuracy_mean", "—")
            acc_std  = stats.get("accuracy_std",  "—")
            f1_mean  = stats.get("macro_f1_mean", "—")
            f1_std   = stats.get("macro_f1_std",  "—")
            if isinstance(acc_mean, float):
                acc_str = f"{acc_mean:.4f} ± {acc_std:.4f}"
                f1_str  = f"{f1_mean:.4f} ± {f1_std:.4f}"
            else:
                acc_str = f1_str = "—"
            ms_rows.append([model, acc_str, f1_str])
        print_table(["Model", "Acc mean ± std", "MacF1 mean ± std"],
                    ms_rows, col_widths=[28, 20, 20])

    # ── McNemar ──────────────────────────────────────────────────────────────
    mn_path = results_dir / "mcnemar_tests.json"
    if mn_path.exists():
        mn = load_json(mn_path)
        print("\n── McNemar Significance Tests (Prompt vs No-Prompt) ──")
        mn_rows = []
        for model, result in mn.items():
            delta = result.get("delta_acc", "—")
            p     = result.get("p_value",   "—")
            sig   = result.get("significant", False)
            delta_str = f"+{delta:.4f}" if isinstance(delta, float) and delta >= 0 else f"{delta:.4f}" if isinstance(delta, float) else "—"
            p_str     = f"{p:.4f}" if isinstance(p, float) else "—"
            sig_str   = "YES *" if sig else "ns"
            mn_rows.append([model, delta_str, p_str, sig_str])
        print_table(["Model", "Δ Acc", "p-value", "Significant"],
                    mn_rows, col_widths=[20, 10, 10, 14])

    # ── Few-shot ──────────────────────────────────────────────────────────────
    fs_path = results_dir / "fewshot_results.json"
    if fs_path.exists():
        fs = load_json(fs_path)
        print("\n── Few-Shot Results (DistilRoBERTa + Prompt) ──")
        fs_rows = []
        for k_str, m in fs.items():
            k = int(k_str)
            n = k * 6
            fs_rows.append([f"{k}-shot/class", f"{n:,}",
                             fmt(m.get("accuracy")), fmt(m.get("macro_f1"))])
        fs_rows.append(["Full fine-tune", "16,000", "0.9260", "0.8829"])
        print_table(["Condition", "Train N", "Acc", "Mac-F1"],
                    fs_rows, col_widths=[22, 10, 8, 8])

    # ── Rebalancing ───────────────────────────────────────────────────────────
    wl_path = results_dir / "weighted_loss_results.json"
    os_path = results_dir / "oversample_results.json"
    if wl_path.exists() or os_path.exists():
        print("\n── Rebalancing Comparison ──")
        rb_rows = []
        for mname in ["ELECTRA-base", "DistilRoBERTa"]:
            std_path = results_dir / f"{mname}_prompt_metrics.json"
            wgt_path = results_dir / f"{mname}_weighted_metrics.json"
            ovs_path = results_dir / f"{mname}_oversample_metrics.json"
            for label, path in [("Standard", std_path), ("Weighted", wgt_path), ("Oversample", ovs_path)]:
                if path.exists():
                    m = load_json(path)
                    rb_rows.append([mname, label,
                                    fmt(m.get("accuracy")), fmt(m.get("macro_f1"))])
                else:
                    rb_rows.append([mname, label, "—", "—"])
        print_table(["Model", "Strategy", "Acc", "Mac-F1"],
                    rb_rows, col_widths=[18, 12, 8, 8])

    # ── File inventory ────────────────────────────────────────────────────────
    print(f"\n── Result Files in {results_dir.name}/ ──")
    all_json = sorted(results_dir.glob("*.json"))
    for p in all_json:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<55} {size_kb:>7.1f} KB")
    print(f"\nTotal: {len(all_json)} JSON files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise experiment results.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/content/drive/MyDrive/emotion_experiment/results"),
        help="Path to results directory (default: Google Drive path)"
    )
    args = parser.parse_args()
    main(args.results_dir)
