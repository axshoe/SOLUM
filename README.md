# SOLUM

**A systematic audit of soil organic carbon prediction model transferability across land use classes using the LUCAS Topsoil dataset.**

SOLUM (from the pedological term for the upper soil horizon where most organic activity occurs) produces the first published Transferability Matrix quantifying how Vis-NIR spectroscopy models trained on one land use class perform when applied to another — and which spectral wavelengths are responsible when they fail.

**Part of [The Xiu Lab](https://thexiulab.org) research portfolio.**  
GitHub: [github.com/axshoe](https://github.com/axshoe)

---

## The question

When a machine learning model for predicting soil organic carbon (SOC) is trained on cropland spectra and then evaluated on woodland spectra, how badly does it break? Which spectral bands cause the failure? Which land use class makes the most "exportable" model?

No published work has answered this systematically for all LUCAS land use classes in a single unified benchmark. That is the gap SOLUM fills.

---

## Methods

| Component | Approach |
|-----------|----------|
| Dataset | LUCAS Topsoil 2018 (EU JRC, ~20,000 georeferenced soil samples, 25 EU member states) |
| Spectral preprocessing | Savitzky-Golay smoothing + SNV (from scratch) |
| Model 1 | PLSR via NIPALS algorithm (from scratch) |
| Model 2 | Random Forest (scikit-learn) with grid search |
| Evaluation | R², RMSE, RPD for all N×N source-target land use pairs |
| Attribution | SHAP wavelength discrepancy for failed transfers (RPD < 1.4) |

Land use classes (LUCAS major codes): A (Arable Cropland), B (Permanent Cropland), C (Grassland), D (Woodland), E (Shrubland), F (Bare Land).

RPD thresholds: ≥2.0 = Good, 1.4–2.0 = Moderate, <1.4 = Poor.

---

## Primary artifact

The Transferability Matrix: an N×N table where entry (source, target) = RPD when a model trained on `source` is evaluated on `target`. Diagonal = in-domain performance. Off-diagonal = transfer performance.

---

## Novelty

The methods (PLSR, RF, SHAP) are standard. The contribution is measurement scope: a systematic, all-pairs benchmark of cross-land-use SOC model transferability, with wavelength-level attribution of where the domain shift manifests spectrally. This is a measurement paper, not a methods paper.

---

## Setup

See [SETUP.md](SETUP.md) for full installation and usage instructions.

Quick start:
```bash
pip install -r requirements.txt
python main.py --lucas path/to/LUCAS_Topsoil_2018.csv --verbose
```

The LUCAS dataset requires free registration at:  
https://esdac.jrc.ec.europa.eu/projects/lucas

---

## Outputs

- `outputs/transferability_matrix_plsr_rpd.csv` — primary artifact
- `outputs/transferability_matrix_rf_rpd.csv`
- `outputs/results_all_pairs.csv`
- `outputs/shap_discrepancy.csv`
- `figures/` — 8 publication-quality figures

---

## Limitations

- Results are specific to the LUCAS geographic coverage (EU member states, 2018 survey)
- Transfer performance will vary under different soil taxonomy regions
- SHAP attributions are approximate (TreeExplainer with path-dependent perturbation)
- PLSR from scratch is slower than sklearn's optimized implementation; not recommended for production deployment

---

## Citation

If you use SOLUM in your work, please cite:

> Xiu, A. (2025). SOLUM: A Cross-Land-Use Transferability Benchmark for Soil Organic Carbon Vis-NIR Prediction Models. The Xiu Lab. https://github.com/axshoe/solum

---

## License

MIT License. See LICENSE.

---

*The Xiu Lab · thexiulab.org · github.com/axshoe*
