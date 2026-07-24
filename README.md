# What Actually Drives Profit in Telecom Churn Retention

Code and experiment notebooks for the paper:

> **What Actually Drives Profit in Telecom Churn Retention: Profit-Aligned Targeting versus Accuracy and Calibration**
> Mumtaheena Binte Ahmed, K. M. Zawad Monsur
> *11th IEEE Asia-Pacific Conference on Computer Science and Data Engineering (CSDE 2026)*

---

## Overview

We ask which components of a churn pipeline actually move **realised retention profit**, and test each effect with bootstrap confidence intervals rather than reporting a single bundled pipeline.

**Headline finding.** Aligning the decision objective with profit significantly beats accuracy-optimal ($F_1$) thresholding when customer lifetime value is heterogeneous and the classifier carries usable signal (Maven: **+13.1 pp** of profit capture, 95% CI [6.7, 19.1]). Class re-sampling does not improve ranking but degrades calibration by up to ~10× in ECE; post-hoc calibration restores it without changing AUC. Once a profit-optimal threshold is in place, neither per-customer refinement nor calibration adds a statistically significant profit gain.

---

## Datasets

Both are public on Kaggle and are **not redistributed here** — add them as inputs in your own notebook environment.

| Dataset | Kaggle slug | Records used | Churn rate |
|---|---|---|---|
| Maven Telecom | `shilongzhuang/telecom-customer-churn-by-maven-analytics` | 6,589 | 28.4% |
| Cell2Cell | `jpacse/datasets-for-churn-telecom` (`cell2celltrain.csv`) | 51,047 | 28.8% |

Maven's target is derived from `Customer Status`: *Churned* vs *Stayed*, with *Joined* (new customers, no outcome) dropped. `Churn Category` / `Churn Reason` leak the label and are excluded from features — they are used only to qualitatively validate SHAP drivers.

---

## Repository layout

```
notebooks/    01–06, run in order
results/      all result CSVs produced by the notebooks
figures/      figures used in the paper
```

---

## Reproduction

Notebooks were run on Kaggle (CPU only — no GPU required). Run them in order; each writes its outputs to `/kaggle/working`, which the next notebook reads.

| Notebook | Produces |
|---|---|
| `01_eda_preprocessing` | Stratified 60/20/20 train/calibration/test splits, feature metadata |
| `02_baseline_models` | Baseline models + metrics, saved probabilities (`results_baseline.csv`, `results_cv_auc.csv`) |
| `03_imbalance_ablation` | Natural vs class-weight vs SMOTE (`results_imbalance.csv`) |
| `04_calibration` | Platt + isotonic, before/after Brier/ECE (`results_calibration.csv`) |
| `05_profit_framework` | Profit policies, bootstrap CIs, sensitivity, Fig. 2 (`results_profit*.csv`) |
| `06_shap_explainability` | Global/local SHAP figures, churn-reason validation |

**If running notebooks separately:** set `INPUT_DIR` / `CPROB_DIR` / `SPLIT_DIR` at the top of each notebook to the committed output path of the notebook it depends on. Running everything in a single session with `INPUT_DIR = "/kaggle/working"` is simplest.

### Protocol

- Split: stratified **60/20/20**, `seed = 42`
- **train** → fit model; **calibration** → fit calibrator and tune all thresholds; **test** → evaluated once
- All statistical transforms (imputation, encoding, scaling) fit on train only; SMOTE applied inside the pipeline to the training fold only

### Profit framework parameters

| Parameter | Value | Meaning |
|---|---|---|
| `margin` (μ) | 0.30 | profit margin on revenue |
| `tenure clip` | [6, 60] months | expected remaining tenure bounds |
| `delta` (δ) | 0.10 | retention offer cost as fraction of CLV |
| `contact cost` (c) | 2.0 (fixed) | per-contact cost; deliberately **not** proportional to CLV |
| `gamma` (γ) | ≈ 0.50 | offer acceptance, estimated empirically from Cell2Cell `RetentionOffersAccepted / RetentionCalls` |

CLV_i = μ · max(monthly_i, 0) · clip(tenure_i, 6, 60)

Target customer *i* iff  EP_i = −c + p_i·γ·CLV_i·(1−δ) − (1−p_i)·δ·CLV_i > 0

Sensitivity analysis varies each of these; conclusions are stable except with respect to δ, to which campaign profitability is naturally sensitive.

---

## Environment

```bash
pip install -r requirements.txt
```

CPU is sufficient; the largest dataset is ~51k rows.

---

## Limitations

- Results come from a **single** train/calibration/test split; the reported confidence intervals capture test-set sampling, not split variability. Multi-seed evaluation is future work.
- Lifetime values are proxies derived from monthly charge and tenure, not measured CLV.
- The acceptance rate estimated from Cell2Cell is transferred to Maven.

---

## Citation

```bibtex
@inproceedings{ahmed2026profit,
  title     = {What Actually Drives Profit in Telecom Churn Retention:
               Profit-Aligned Targeting versus Accuracy and Calibration},
  author    = {Ahmed, Mumtaheena Binte and Monsur, K. M. Zawad},
  booktitle = {11th IEEE Asia-Pacific Conference on Computer Science
               and Data Engineering (CSDE)},
  year      = {2026}
}
```
