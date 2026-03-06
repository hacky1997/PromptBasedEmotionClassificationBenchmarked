# Figures

All figures are generated automatically by **Cell 10** (in-domain) and **Cell 15** (cross-domain) of the notebook. They are saved to `MyDrive/emotion_experiment/figures/` as both PDF (vector, for the paper) and PNG (300 DPI, for presentations).

## In-Domain Figures (Cell 10)

| Filename | Description | Paper Reference |
|---|---|---|
| `fig1_accuracy_macro_f1.pdf/png` | Grouped bar chart: Acc and Macro-F1 across all 9 systems | Figure 1 |
| `fig2_efficiency_scatter.pdf/png` | Accuracy vs. throughput Pareto frontier (log scale) | Figure 2 |
| `fig3_prompt_ablation.pdf/png` | Prompt vs. no-prompt accuracy deltas with significance markers | Figure 3 |
| `fig4_confusion_matrices.pdf/png` | Normalised confusion matrices for DistilBERT and DistilRoBERTa | Figure 4 |
| `fig5_perclass_heatmap.pdf/png` | Per-class F1 heatmap across all five transformer models | Figure 5 |
| `fig6_label_distribution.pdf/png` | Training split class distribution bar chart | Figure 6 |

## Cross-Domain Figures (Cell 15)

| Filename | Description | Paper Reference |
|---|---|---|
| `fig7_crossdomain_bar.pdf/png` | In-domain vs. cross-domain Acc and Macro-F1 comparison | Figure 7 |
| `fig8_crossdomain_f1_heatmap.pdf/png` | Per-class F1 heatmap across three evaluation domains | Figure 8 |
| `fig9_crossdomain_confusion.pdf/png` | Confusion matrices for GoEmotions (strict) and MELD | Figure 9 |

## Notes

- All figures use publication-quality settings: 300 DPI PNG, vector PDF, DejaVu Sans font, no top/right spines, 0.3 alpha grid.
- Figures are **not committed to this repository** (PNG/PDF excluded by `.gitignore`) to keep the repo lightweight.
- To regenerate figures, run the notebook end-to-end or from Cell 9 onward (results must already be on Drive).
