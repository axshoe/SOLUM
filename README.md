# SOLUM

**A systematic benchmark of soil organic carbon prediction model transferability across land use classes, using the LUCAS Topsoil dataset with both VNIR and MIR spectroscopy.**

SOLUM produces a Transferability Matrix quantifying how soil organic carbon (SOC) spectroscopy models trained on one land use class perform when applied to another, and diagnoses why they fail: through representativeness analysis, an independent mid-infrared replication, and a covariate check. The headline result is that cropland is spectrally isolated. No model trained on any other land use class can predict cropland SOC, and the failure survives a higher-fidelity instrument.

**Part of [The Xiu Lab](https://thexiulab.org) research portfolio.**
GitHub: [github.com/axshoe](https://github.com/axshoe)

---

## The question

When a machine learning model for predicting SOC is trained on woodland spectra and then applied to cropland spectra, how badly does it break? Which land use class produces the most transferable model? And when a transfer fails, is it because the target soil is unfamiliar to the model (a distribution problem that better data could fix), or because the relationship between spectrum and carbon is genuinely different there (a problem no amount of same-class data can fix)?

No published work had answered these systematically for the LUCAS land use classes in a single unified benchmark. That is the gap SOLUM fills.

---

## Methods

| Component | Approach |
|-----------|----------|
| Primary dataset | LUCAS 2015 Topsoil (EU JRC), 21,677 samples after filtering, five land use classes |
| Independent replication | LUCAS MIR subset from the Open Soil Spectral Library (OSSL) |
| Spectral preprocessing | Savitzky-Golay smoothing and SNV, implemented from scratch |
| Model 1 | PLSR via the NIPALS algorithm, implemented from scratch |
| Model 2 | Random Forest (scikit-learn) with grid search |
| Evaluation | R squared, RMSE, and RPD for all ordered source-target land use pairs |
| Pooled baseline | A single all-class model compared against per-class models |
| Stability | Five repeated holdouts with different random seeds |
| Mechanism | Representativeness analysis (PLS score projection, Mahalanobis distance) separating covariate shift from concept shift |
| Attribution | SHAP wavelength discrepancy for failed transfers (RPD < 1.4) |
| Covariate check | Correlation of ancillary soil properties with the visible-range discrepancy |

Land use classes follow the LUCAS LC1 nomenclature (first letter of the code): B (Cropland, predominantly arable annual crops with a minority of permanent woody crops), C (Woodland), D (Shrubland), E (Grassland), F (Bare Land). Artificial land (A), water (G), and wetland (H) are not agricultural soil and are excluded.

RPD thresholds: 2.0 or above is Good, 1.4 to 2.0 is Moderate, below 1.4 is Poor. RPD of 1.0 is the mean-predictor baseline.

---

## Primary artifact

The Transferability Matrix: a table where entry (source, target) is the RPD when a model trained on the source class is evaluated on the target class. The diagonal is in-domain performance; off-diagonal entries are transfer performance.

---

## Key findings

- Cropland is spectrally isolated. Every non-cropland source class produces RF RPD below 1.4 when predicting cropland, and woodland-to-cropland and shrubland-to-cropland fall below the mean-predictor baseline of 1.0.
- The failure is a concept shift, not a distribution problem. Cropland test samples fall inside the source models' spectral score distributions (for example, shrubland-to-cropland is 91.7% within distribution) yet still fail, so the models are interpolating and still wrong.
- The failure persists under mid-infrared spectroscopy and intensifies (woodland-to-cropland RPD 0.31), so higher spectral fidelity does not resolve it.
- Ancillary soil covariates (clay, pH, texture, EC) show no meaningful correlation with the visible-range discrepancy, ruling out a simple mineralogy explanation.
- Pooling all classes into one model, the standard practice, improves most classes while degrading cropland by 0.30 RPD, a trade-off invisible in aggregate metrics.

---

## Novelty

The methods (PLSR, RF, SHAP) are standard. The contribution is measurement and diagnosis: a systematic all-pairs benchmark of cross-land-use SOC transferability, a separation of covariate shift from concept shift, and an independent cross-instrument replication. This is a measurement paper, not a methods paper.

---

## Setup

See [SETUP.md](SETUP.md) for full installation and usage instructions.

Quick start:

```bash
pip install -r requirements.txt
python solum/main.py --chemistry data/LUCAS_Topsoil_2015_20200323.csv --spectra data/spectra/ --verbose
```

The LUCAS dataset requires free registration at:
https://esdac.jrc.ec.europa.eu/projects/lucas

The OSSL data used for the MIR replication is available through:
https://soilspectroscopy.org

---

## Outputs

- `outputs/transferability_matrix_plsr_rpd.csv` and `outputs/transferability_matrix_rf_rpd.csv` — primary artifacts
- `outputs/pooled_baseline_results.csv` — pooled versus per-class comparison
- `outputs/transferability_matrix_rf_mean.csv` and `_sd.csv` — repeated holdout stability
- `outputs/representativeness_mahal.csv` — covariate versus concept shift diagnosis
- `outputs/log1p_evaluation_full.csv` — dual-scale evaluation
- `outputs/covariate_correlations.csv` — ancillary covariate check
- `outputs/transferability_matrix_mir_plsr_rpd.csv` — MIR replication
- `figures/` — publication-quality figures

---

## Limitations

- Results reflect the LUCAS geographic coverage (EU member states). Whether the magnitudes transfer to other continents is untested.
- The cropland class pools arable and permanent woody crops under the single LUCAS class B. The arable subset dominates, but a finer arable-versus-permanent split is a natural extension using the LUCAS 2018 survey, which resolves crop type in more detail.
- The MIR replication is limited to three classes (cropland, woodland, grassland) because the archive subset was thin on shrubland and bare land.
- SHAP attributions are approximate (TreeExplainer, path-dependent perturbation) and identify where models diverge spectrally, not the underlying chemistry.
- The mineral-associated versus particulate organic matter mechanism is a hypothesis, not a demonstrated cause.

---

## Citation

If you use SOLUM in your work, please cite:

> Xiu, A. (2026). Cropland is spectrally isolated for soil organic carbon prediction: a cross-instrument transferability benchmark on LUCAS with VNIR and MIR. The Xiu Lab. https://github.com/axshoe/SOLUM

---

## License

MIT License. See LICENSE.

---

*The Xiu Lab · thexiulab.org · github.com/axshoe*
