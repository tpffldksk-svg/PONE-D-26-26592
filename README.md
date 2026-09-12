[README.md](https://github.com/user-attachments/files/32142414/README.md)
# Firth Penalized Logistic Regression \u2014 RBS Prediction Model

Statistical analysis code for the manuscript:

**"Predicting Red Breast Syndrome After ADM-Assisted Breast Reconstruction:
An Exploratory Scoring Model Using Routine Hematologic Markers"**
(PONE-D-26-26592)

## Primary analysis (current)

`firth_regression_validated.R` \u2014 The validated, final analysis pipeline,
using the [`logistf`](https://cran.r-project.org/package=logistf) R package
(Heinze & Ploner), a peer-reviewed, actively maintained implementation of
Firth's penalized logistic regression method authored by the original
developers of the method itself. This script performs:

1. Backward elimination via `logistf::backward()`
2. Composite hematologic combination score construction and ROC analysis
3. Bootstrap internal validation (2,000 resamples)
4. Model calibration (calibration slope, calibration-in-the-large) and
   Brier score

All Firth regression estimates, profile penalized likelihood confidence
intervals, and likelihood ratio test p-values reported in the manuscript
were computed using this script, ensuring internally consistent
inferential procedures throughout.

### Dependencies
- R (>= 4.3.0)
- `readxl`
- `logistf` (v1.26.1 or later)
- `pROC`

### Usage
```r
source("firth_regression_validated.R")
```
Update the `read_excel()` path at the top of the script to point to your
local copy of the study dataset.

## Superseded analysis (archival)

`firth_regression.py` \u2014 An earlier custom Python implementation of Firth's
method, retained here for transparency and reproducibility of the original
analysis history. Following editorial review, this implementation was
independently cross-checked against `logistf`, and the R-based results in
`firth_regression_validated.R` are reported as the primary and final
analysis in the published manuscript.

## License

MIT License (see `LICENSE`).
