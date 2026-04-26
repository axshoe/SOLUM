# SOLUM — Setup Instructions

**Open in: PyCharm**

---

## Prerequisites

- Python 3.10 or later
- PyCharm (Community or Professional)
- Git

---

## 1. Clone the repository

```bash
git clone https://github.com/axshoe/solum.git
cd solum
```

---

## 2. Create a virtual environment

In PyCharm:
- Open the project folder
- Go to **File > Settings > Project: solum > Python Interpreter**
- Click the gear icon > **Add Interpreter > Add Local Interpreter > Virtualenv**
- Set base interpreter to your Python 3.10+ installation
- Click OK

Or from terminal:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, shap.

---

## 4. Obtain the LUCAS Topsoil dataset

SOLUM uses the **LUCAS 2018 Topsoil** dataset, which is free but requires
registration with the EU Joint Research Centre (JRC).

1. Go to: https://esdac.jrc.ec.europa.eu/projects/lucas
2. Register for a free account (institutional or personal email accepted)
3. Download the 2018 LUCAS Topsoil CSV file
4. Place the file anywhere accessible, e.g.:

```
solum/
  data/
    LUCAS_Topsoil_2018.csv   ← place here
```

The 2009/2012 LUCAS dataset is also supported for secondary validation and
is downloadable from the same JRC page.

---

## 5. Run the pipeline

```bash
cd solum
python main.py --lucas data/LUCAS_Topsoil_2018.csv --verbose
```

### Common flags

| Flag | Description |
|------|-------------|
| `--lucas PATH` | Path to LUCAS CSV (required) |
| `--no-grid-search` | Skip RF hyperparameter tuning (faster; uses defaults) |
| `--no-shap` | Skip SHAP computation (much faster; no attribution figures) |
| `--sample-n N` | Use only N samples per class (e.g., `--sample-n 200` for quick testing) |
| `--output-dir DIR` | Directory for CSV outputs (default: `outputs/`) |
| `--random-state INT` | Random seed (default: 42) |
| `--verbose` | Print detailed progress |

### Quick test run (no GPU required, ~5 minutes)

```bash
python main.py --lucas data/LUCAS_Topsoil_2018.csv --sample-n 200 --no-grid-search --no-shap --verbose
```

### Full run (recommended, ~30-90 min depending on CPU)

```bash
python main.py --lucas data/LUCAS_Topsoil_2018.csv --verbose
```

---

## 6. Outputs

After a successful run, you will find:

```
outputs/
  transferability_matrix_plsr_rpd.csv   ← primary artifact (PLSR)
  transferability_matrix_rf_rpd.csv     ← primary artifact (RF)
  results_all_pairs.csv                 ← full R², RMSE, RPD for all pairs
  shap_discrepancy.csv                  ← wavelength attribution (if SHAP enabled)

figures/
  fig1_soc_distribution.png
  fig2_mean_spectra.png
  fig3_indomain_plsr.png
  fig4_indomain_rf.png
  fig5_transferability_plsr.png
  fig6_transferability_rf.png
  fig7_shap_discrepancy.png
  fig8_top_discrepant_bands.png
```

---

## 7. Column name notes

LUCAS CSV column names vary slightly between dataset versions. If you get a
`KeyError` for the SOC or land use column, open `data_loader.py` and update:

```python
SOC_COL = "OC"     # or "soc", "SOC", "org_c" — check your CSV header
LU_COL  = "LC1"    # or "LC0", "land_cover" — check your CSV header
```

The spectral band columns (wavelengths) are auto-detected.

---

## 8. Project structure

```
solum/
  main.py                     — orchestration
  data_loader.py              — LUCAS loading and validation
  spectral_preprocessing.py   — SG smoothing + SNV (from scratch)
  plsr.py                     — NIPALS PLSR (from scratch)
  rf_model.py                 — Random Forest wrapper (scikit-learn)
  transferability_matrix.py   — cross-evaluation loop
  shap_analysis.py            — SHAP attribution
  analysis.py                 — figure generation
requirements.txt
SETUP.md
README.md
```

---

## 9. PLSR design note

PLSR is implemented from scratch (NIPALS algorithm in `plsr.py`) rather than
using `sklearn.cross_decomposition.PLSRegression`. This is a deliberate design
choice: working through NIPALS explicitly exposes how latent variables are
extracted, why deflation works, and what the regression coefficients actually
represent in spectral space. For production deployment, sklearn's optimized
implementation would be a reasonable swap.

---

## 10. GitHub

```
https://github.com/axshoe/solum
```

See also: [thexiulab.org](https://thexiulab.org)
