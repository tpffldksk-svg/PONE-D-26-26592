#############################################################################
# Validated Firth Penalized Logistic Regression Analysis
#############################################################################
# Manuscript: "Predicting Red Breast Syndrome After ADM-Assisted Breast
# Reconstruction: An Exploratory Scoring Model Using Routine Hematologic
# Markers" (PONE-D-26-26592)
#
# This script performs the FINAL, validated statistical analysis using the
# logistf R package (Heinze & Ploner), a peer-reviewed, actively maintained
# implementation of Firth's penalized logistic regression method authored
# by the original developers of the method itself.
#
# This script supersedes an earlier custom Python implementation. Following
# editorial review, all Firth regression estimates, profile penalized
# likelihood confidence intervals, and likelihood ratio test p-values were
# independently re-verified using logistf, and the logistf-derived results
# are reported as the primary and final analysis throughout the manuscript.
#
# Dependencies: readxl, logistf, pROC
# R version: 4.3.3
# logistf version: 1.26.1
#############################################################################

library(readxl)
library(logistf)
library(pROC)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# Replace with the path to the study dataset (ADM cohort, n = 47)
df <- read_excel("data.xlsx", sheet = 1)

rbs        <- df[[6]]                                  # Red breast syndrome (0/1)
wbc        <- df[[36]]                                 # WBC (x10^3/uL)
plt        <- df[[44]]                                 # Platelet count (x10^3/uL)
hb         <- df[[38]]                                 # Hemoglobin (g/dL)
ki67_pre   <- df[[33]]                                 # Pre-NAC Ki-67 (1=Low, 2=Intermediate, 3=High)
ki67_high  <- as.numeric(ki67_pre == 3)
baso       <- df[[51]]                                 # Basophil (%)
n_stage    <- as.numeric(df[[63]] >= 1)                # ypN >= 1
tils       <- df[[34]]                                 # Stromal TILs (%)

cat("N =", length(rbs), ", RBS events =", sum(rbs), "\n\n")

# ---------------------------------------------------------------------------
# 2. Backward elimination (validated via logistf::backward)
# ---------------------------------------------------------------------------
dat <- data.frame(rbs = rbs, wbc = wbc, hb = hb, plt = plt,
                   baso = baso, n_stage = n_stage, ki67_high = ki67_high, tils = tils)

# Candidate variables entering at p<0.20 in univariable screening
# (ANC excluded a priori due to collinearity with WBC)
fit_full <- logistf(rbs ~ wbc + hb + plt + baso + n_stage + ki67_high + tils,
                     data = dat, firth = TRUE, pl = TRUE)

cat("=== Backward elimination (removal threshold p > 0.10) ===\n")
fit_final <- backward(fit_full, data = dat, trace = TRUE, slstay = 0.10)

cat("\n=== FINAL MODEL ===\n")
res <- data.frame(
  OR       = exp(coef(fit_final)),
  CI_lower = exp(fit_final$ci.lower),
  CI_upper = exp(fit_final$ci.upper),
  p_value  = fit_final$prob
)
print(res, digits = 4)

# ---------------------------------------------------------------------------
# 3. Composite hematologic combination score (WBC + PLT + Ki-67 High)
# ---------------------------------------------------------------------------
wbc_cutoff <- 9.45   # Youden Index-derived cutoff
plt_cutoff <- 177    # Youden Index-derived cutoff

score <- as.numeric(wbc >= wbc_cutoff) + as.numeric(plt >= plt_cutoff) + ki67_high

roc_score <- roc(rbs, score, quiet = TRUE)
cat("\nComposite score AUC =", as.numeric(auc(roc_score)), "\n")
print(ci.auc(roc_score, method = "delong"))

best <- coords(roc_score, "best", best.method = "youden",
               ret = c("threshold", "sensitivity", "specificity", "ppv", "npv"))
cat("\nOptimal threshold (Youden):\n")
print(best)

# ---------------------------------------------------------------------------
# 4. Bootstrap internal validation (2,000 resamples)
# ---------------------------------------------------------------------------
set.seed(42)
n <- length(rbs)
n_boot <- 2000
train_aucs <- c(); test_aucs <- c()

for (b in 1:n_boot) {
  idx <- sample(1:n, n, replace = TRUE)
  oob <- setdiff(1:n, idx)
  if (length(oob) < 3) next
  y_tr <- rbs[idx]; y_oob <- rbs[oob]
  if (length(unique(y_tr)) < 2 || length(unique(y_oob)) < 2) next
  s_tr <- score[idx]; s_oob <- score[oob]
  if (length(unique(s_tr)) < 2) next

  auc_tr  <- tryCatch(as.numeric(auc(roc(y_tr, s_tr, quiet = TRUE))), error = function(e) NA)
  auc_oob <- tryCatch(as.numeric(auc(roc(y_oob, s_oob, quiet = TRUE))), error = function(e) NA)
  if (is.na(auc_tr) || is.na(auc_oob)) next

  train_aucs <- c(train_aucs, auc_tr)
  test_aucs  <- c(test_aucs, auc_oob)
}

apparent_auc  <- as.numeric(auc(roc_score))
optimism      <- mean(train_aucs - test_aucs)
corrected_auc <- apparent_auc - optimism

cat("\n=== Bootstrap Internal Validation (n=2,000) ===\n")
cat("Apparent AUC:", round(apparent_auc, 3), "\n")
cat("Mean train AUC:", round(mean(train_aucs), 3), " SD:", round(sd(train_aucs), 3), "\n")
cat("Mean OOB test AUC:", round(mean(test_aucs), 3), " SD:", round(sd(test_aucs), 3), "\n")
cat("Optimism:", round(optimism, 4), "\n")
cat("Bootstrap-corrected AUC:", round(corrected_auc, 3), "\n")
cat("95% CI (OOB percentile):", round(quantile(test_aucs, 0.025), 3), "-",
    round(quantile(test_aucs, 0.975), 3), "\n")

# ---------------------------------------------------------------------------
# 5. Calibration and Brier score (final multivariable model)
# ---------------------------------------------------------------------------
mu <- predict(fit_final, type = "response")

logit_mu <- log(mu / (1 - mu))
cal_fit <- logistf(rbs ~ logit_mu, firth = TRUE, pl = TRUE)

cat("\n=== Calibration ===\n")
cat("Calibration-in-the-large:", round(coef(cal_fit)[1], 3), "\n")
cat("Calibration slope:", round(coef(cal_fit)[2], 3), "\n")

brier      <- mean((rbs - mu)^2)
brier_null <- mean((rbs - mean(rbs))^2)
cat("Brier score:", round(brier, 3), " (null:", round(brier_null, 3),
    ", scaled:", round(1 - brier/brier_null, 3), ")\n")
