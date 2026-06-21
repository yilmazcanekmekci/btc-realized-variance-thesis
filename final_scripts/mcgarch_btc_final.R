# ARMA(1,0)-GARCH(1,1) on log(z_t) 
#   RV_{t,i} = h_t · s_i · q_{t,i}
#   h_t : previous day's mean hourly RV (low-freq component)
#   s_i : diurnal component (mean RV/h_prev per hour-of-day)
#   q_{t,i}: modelled by ARMA(1,0)-GARCH(1,1) on log(z) where z = RV/(h·s)

# DATA WINDOW: 2016-12-01T14:30 → 2025-12-31T23:55 (aligned with Python)
# SPLIT: Train ≤2021-12-31 / Val 2022-2023 / Test 2024-2025

# OUTPUTS (upload all three to kaggle notebook):
# - mcgarch_forecasts.csv datetime_utc, log_RV_hat, sigma2_t
# - mcgarch_wfv.csv — fold metrics
# - mcgarch_metadata.csvGARCH parameters + resid_var

# * As I am not fully comfortable in R programming syntax and code 
# structure, some part of section-4 and section-5  of the code development, debugging, and formatting
# process were assisted by(LLMs). All generated code was 
# reviewed, adapted, and validated within the context of this study.

library(data.table)
library(rugarch)
library(lubridate)

# Configuration
CSV_PATH<- "C:/Users/YILMAZ/Desktop/BTCUSD_5min_2015_2025.csv"
OUTPUT_DIR <- "C:/Users/YILMAZ/Desktop"

# Data window — aligned with Python btc_thesis_v3.py
DATA_START <- as.POSIXct("2016-12-01 14:30:00", tz = "UTC")
DATA_END<- as.POSIXct("2025-12-31 23:55:00", tz = "UTC")

TRAIN_END <- as.POSIXct("2021-12-31 23:59:59", tz = "UTC")
VAL_END  <- as.POSIXct("2023-12-31 23:59:59", tz = "UTC")
WFV_N_SPLITS <- 5L

cat("  MC-GARCH v3 — Bitcoin Hourly Realized Volatility\n")
cat(sprintf(" Data window: %s → %s\n", DATA_START, DATA_END))
cat(sprintf(" Split: Train ≤%s  Val ≤%s  Test: rest\n",as.Date(TRAIN_END), as.Date(VAL_END)))

# SECTION 1 — LOAD & FILTER DATA

cat("\n[1] Loading and filtering data...\n")

dt <- fread(CSV_PATH)
dt[, timestamp := as.POSIXct(timestamp, tz = "UTC")]
setkey(dt, timestamp)

# Apply data window
dt <- dt[timestamp >= DATA_START & timestamp <= DATA_END]
cat(sprintf("Rows after window filter : %d\n", nrow(dt)))
cat(sprintf("Range: %s → %s\n", min(dt$timestamp), max(dt$timestamp)))


# SECTION 2 — HOURLY REALIZED VOLATILITY

cat("\n[2] Computing hourly RV...\n")

dt[, hour := floor_date(timestamp, "hour")]
dt[, log_close := log(close)]
dt[, r_5m := log_close - shift(log_close), by = hour]
dt[, r2_5m:= r_5m^2]

# Bars-per-hour check
n_per_h <- dt[, .N, by = hour]
cat(sprintf(" Bars-per-hour: min=%d  median=%d  max=%d\n", min(n_per_h$N), as.integer(median(n_per_h$N)), max(n_per_h$N)))

rv_hourly <- dt[, .(
  RV= sum(r2_5m, na.rm = TRUE),
  n_obs = sum(!is.na(r2_5m))
), by = hour]
rv_hourly[, log_RV := log(ifelse(RV > 0, RV, NA_real_))]
setkey(rv_hourly, hour)

cat(sprintf(" Total hourly obs : %d\n",   nrow(rv_hourly)))
cat(sprintf(" Zero-RV hours    : %d\n",   sum(rv_hourly$RV == 0)))
cat(sprintf(" log_RV range     : [%.2f, %.2f]\n", min(rv_hourly$log_RV, na.rm = TRUE), max(rv_hourly$log_RV, na.rm = TRUE)))


# SECTION 3 — TRAIN / VAL / TEST SPLIT

cat("\n[3] Splitting data...\n")

rv_train <- rv_hourly[hour <= TRAIN_END]
rv_val <- rv_hourly[hour > TRAIN_END & hour <= VAL_END]
rv_test <- rv_hourly[hour > VAL_END]
rv_all  <- rv_hourly[hour <= VAL_END]   # train + val for final fit

cat(sprintf("  %-14s : %6d hours  (%s → %s)\n", "Train",
            nrow(rv_train), as.Date(min(rv_train$hour)),
            as.Date(max(rv_train$hour))))
cat(sprintf("  %-14s : %6d hours  (%s → %s)\n", "Validation",
            nrow(rv_val), as.Date(min(rv_val$hour)),
            as.Date(max(rv_val$hour))))
cat(sprintf("  %-14s : %6d hours  (%s → %s)\n", "Test",
            nrow(rv_test), as.Date(min(rv_test$hour)),
            as.Date(max(rv_test$hour))))
cat(sprintf("  %-14s : %6d hours  (for final fit)\n",
            "Train+Val", nrow(rv_all)))


# SECTION 4 — MC-GARCH FUNCTIONS

cat("\n[4] Defining MC-GARCH decomposition functions...\n")

#  estimate_mcgarch
# Fits MC-GARCH on a training dataset.
# Returns: s_i (diurnal), params (ARMA-GARCH coeffs), fit object.
estimate_mcgarch <- function(rv_data) {
  rv_data <- copy(rv_data[!is.na(RV) & RV > 0])
  
  # Step-1: h_t = prev day's mean hourly RV
  rv_data[, date := as.Date(hour)]
  daily_rv <- rv_data[, .(daily_RV = mean(RV, na.rm = TRUE)), by = date]
  setkey(daily_rv, date)
  daily_rv[, h_prev := shift(daily_RV, 1)]
  rv_data  <- merge(rv_data, daily_rv[, .(date, h_prev)],
                    by = "date", all.x = TRUE)
  rv_data  <- rv_data[!is.na(h_prev) & h_prev > 0]
  
  # Step-2: diurnal s_i per hour-of-day
  rv_data[, y_norm      := RV / h_prev]
  rv_data[, hour_of_day := hour(hour)]
  s_i <- rv_data[, .(s_i = mean(y_norm, na.rm = TRUE)), by = hour_of_day]
  setkey(s_i, hour_of_day)
  rv_data <- merge(rv_data, s_i, by = "hour_of_day", all.x = TRUE)
  
  # Step-3: stochastic component z → log_z
  rv_data[, z     := y_norm / s_i]
  rv_data  <- rv_data[is.finite(z) & z > 0]
  rv_data[, log_z := log(z)]
  
  # Step 4: ARMA(1,0)-GARCH(1,1) on log_z (multi-solver fallback)
  spec <- ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model         = list(armaOrder = c(1, 0), include.mean = FALSE),
    distribution.model = "norm"
  )
  fit <- tryCatch(
    ugarchfit(spec, data = rv_data$log_z, solver = "hybrid",  out.sample = 0),
    error = function(e) tryCatch(
      ugarchfit(spec, data = rv_data$log_z, solver = "nlminb", out.sample = 0),
      error = function(e2)
        ugarchfit(spec, data = rv_data$log_z, solver = "solnp",  out.sample = 0)
    )
  )
  
  p <- coef(fit)
  resid_var <- mean(residuals(fit, standardize = FALSE)^2)
  
  list(
    s_i    = s_i,
    params = list(
      phi   = p["ar1"],
      omega   = p["omega"],
      alpha  = p["alpha1"],
      beta   = p["beta1"],
      persistence = p["alpha1"] + p["beta1"],
      resid_var  = resid_var
    ),
    fit = fit
  )
}

#  mcgarch_forecast_sequence

# One-step-ahead forecasts + per-step GARCH variance (sigma2_t).
# sigma2_t is passed to Python for the Jensen correction:
#   RV_hat = exp(log_RV_hat + 0.5 * sigma2_t)

mcgarch_forecast_sequence <- function(rv_data, s_i_est, params) {
  phi <- params$phi
  omega <- params$omega
  alpha <- params$alpha
  beta  <- params$beta
  
  rv_data <- copy(rv_data[!is.na(RV) & RV > 0])
  rv_data[, date := as.Date(hour)]
  daily_rv <- rv_data[, .(daily_RV = mean(RV, na.rm = TRUE)), by = date]
  setkey(daily_rv, date)
  daily_rv[, h_prev := shift(daily_RV, 1)]
  rv_data  <- merge(rv_data, daily_rv[, .(date, h_prev)],
                    by = "date", all.x = TRUE)
  rv_data[, hour_of_day := hour(hour)]
  rv_data  <- merge(rv_data, s_i_est, by = "hour_of_day", all.x = TRUE)
  rv_data[, y_norm := RV / h_prev]
  rv_data[, z   := y_norm / s_i]
  rv_data[, log_z := ifelse(is.finite(z) & z > 0, log(z), NA_real_)]
  setkey(rv_data, hour)
  
  n   <- nrow(rv_data)
  log_z_v <- rv_data$log_z
  h_v <- rv_data$h_prev
  s_v   <- rv_data$s_i
  hours_v <- rv_data$hour
  preds <- rep(NA_real_, n - 1)
  sigma2_v    <- rep(NA_real_, n - 1)
  # Initialise sigma2 at unconditional variance
  sigma2_prev <- omega / max(1 - alpha - beta, 1e-8)
  
  for (i in seq_len(n - 1)) {
    lz_t <- if (is.na(log_z_v[i]))           0.0 else log_z_v[i]
    prev_lz <- if (i == 1 || is.na(log_z_v[i - 1])) 0.0 else log_z_v[i - 1]
    eps_t   <- lz_t - phi * prev_lz
    
    # GARCH(1,1) one-step-ahead variance update
    sigma2_next <- omega + alpha * eps_t^2 + beta * sigma2_prev
    sigma2_prev <- sigma2_next
    
    h_next <- h_v[i + 1]
    s_next <- s_v[i + 1]
    if (!is.na(h_next) && !is.na(s_next) && h_next > 0 && s_next > 0) {
      # log(RV_hat) = log(h) + log(s) + phi*log(z_t)
      preds[i]    <- log(h_next) + log(s_next) + phi * lz_t
      sigma2_v[i] <- sigma2_next
    }
  }
  
  data.table(datetime_utc = hours_v[-1],
             log_RV_hat   = preds,
             sigma2_t     = sigma2_v)
}

# Helper: evaluate forecasts 
eval_forecasts <- function(preds_dt, actual_dt, label = "") {
  ev <- merge(preds_dt, actual_dt, by = "datetime_utc")
  ev <- ev[is.finite(log_RV_hat) & is.finite(log_RV)]
  e  <- ev$log_RV - ev$log_RV_hat
  rv_t  <- exp(ev$log_RV)
  rv_h  <- pmax(exp(ev$log_RV_hat), 1e-20)
  ratio <- rv_t / rv_h
  qlike <- mean(ratio - log(ratio) - 1)
  if (nchar(label) > 0) {
    cat(sprintf("    %s: n=%d  RMSE=%.4f  MAE=%.4f  QLIKE=%.4f\n",
                label, nrow(ev), sqrt(mean(e^2)), mean(abs(e)), qlike))
  }
  list(rmse = sqrt(mean(e^2)), mae = mean(abs(e)), qlike = qlike, n = nrow(ev))
}

cat("Functions defined.\n")


# SECTION 5 — WALK-FORWARD CV

cat("\n[5] Walk-forward CV (", WFV_N_SPLITS, " folds, expanding window)...\n", sep="")
cat(paste(rep("═", 70), collapse = ""), "\n")

rv_valid  <- rv_all[!is.na(log_RV)]
n_all <- nrow(rv_valid)
min_train <- nrow(rv_train[!is.na(log_RV)])
# WFV_N_SPLITS+1 -> WFV_N_SPLITS: cover all validation period
chunk  <- as.integer((n_all - min_train) / WFV_N_SPLITS)
wfv_rows  <- list()

cat(sprintf("  Total obs: %d | min_train: %d | chunk: %d\n\n", n_all, min_train, chunk))

for (k in seq_len(WFV_N_SPLITS)) {
  tr_end <- min_train + (k - 1L) * chunk
  va_end <- tr_end + chunk
  if (va_end > n_all) break
  
  tr_f <- rv_valid[seq_len(tr_end)]
  va_f <- rv_valid[(tr_end + 1L):va_end]
  cat(sprintf("  Fold %d: train=%d  val=%d  (%s → %s)\n",
              k, nrow(tr_f), nrow(va_f),
              as.Date(min(va_f$hour)), as.Date(max(va_f$hour))))
  
  fit_k <- tryCatch(estimate_mcgarch(tr_f), error = function(e) {
    cat(sprintf("    ✗ Estimation failed: %s\n", e$message)); NULL
  })
  if (is.null(fit_k)) next
  
  preds_k <- tryCatch(
    mcgarch_forecast_sequence(va_f, fit_k$s_i, fit_k$params),
    error = function(e) {
      cat(sprintf("    ✗ Forecast failed: %s\n", e$message)); NULL
    })
  if (is.null(preds_k)) next
  
  act_k <- va_f[!is.na(log_RV), .(datetime_utc = hour, log_RV)]
  ev_k  <- eval_forecasts(preds_k, act_k, label = paste0("fold ", k))
  
  wfv_rows[[k]] <- data.table(
    fold  = k,
    period_from = as.character(as.Date(min(va_f$hour))),
    period_to = as.character(as.Date(max(va_f$hour))),
    train_hrs = nrow(tr_f),
    val_hrs = nrow(va_f),
    mse = ev_k$rmse^2,
    rmse = ev_k$rmse,
    mae = ev_k$mae,
    qlike = ev_k$qlike,
    phi = fit_k$params$phi,
    alpha_beta  = fit_k$params$persistence,
    resid_var = fit_k$params$resid_var
  )
}

wfv_df <- rbindlist(wfv_rows)
cat(paste(rep("─", 70), collapse = ""), "\n")
cat(sprintf("  WFV Mean: RMSE=%.4f  MAE=%.4f  QLIKE=%.4f\n", mean(wfv_df$rmse), mean(wfv_df$mae), mean(wfv_df$qlike)))


# SECTION 6 — FINAL ESTIMATION (TRAIN + VAL)

cat("\n[6] Final estimation on TRAIN + VAL (2017-2023)...\n")

fit_final <- estimate_mcgarch(rv_all)

cat("\n  ARMA(1,0)-GARCH(1,1) estimates:\n")
cat(paste(rep("─", 50), collapse = ""), "\n")
cat(sprintf(" phi (AR1)  : %10.6f\n", fit_final$params$phi))
cat(sprintf(" omega: %10.8f\n", fit_final$params$omega))
cat(sprintf(" alpha (ARCH) : %10.6f\n", fit_final$params$alpha))
cat(sprintf(" beta (GARCH) : %10.6f\n", fit_final$params$beta))
pers <- fit_final$params$persistence
cat(sprintf("  persistence  : %10.6f  (%s)\n", pers,
            ifelse(pers >= 1.0, "INTEGRATED",
                   ifelse(pers > 0.99, "near-integrated", "stable"))))
cat(sprintf("  resid_var    : %10.6f  (σ² for Jensen correction in Python)\n",
            fit_final$params$resid_var))
cat(paste(rep("─", 50), collapse = ""), "\n")

cat("\n  Diurnal factors s_i (by hour-of-day):\n")
print(fit_final$s_i[order(hour_of_day)])

# SECTION 7 — TEST SET FORECASTS (2024-2025)

cat("\n[7] Test set forecasts (2024-2025)...\n")

tp <- mcgarch_forecast_sequence(rv_test, fit_final$s_i, fit_final$params)
at <- rv_test[!is.na(log_RV), .(datetime_utc = hour, log_RV)]
ev <- eval_forecasts(tp, at, label = "test")

# Mincer-Zarnowitz
ev_full <- merge(tp, at, by = "datetime_utc")
ev_full <- ev_full[is.finite(log_RV_hat) & is.finite(log_RV)]
mz_fit  <- lm(log_RV ~ log_RV_hat, data = ev_full)
cat(sprintf("  MZ: alpha=%.4f  beta=%.4f  R²=%.4f\n",
            coef(mz_fit)[1], coef(mz_fit)[2],
            summary(mz_fit)$r.squared))


# SECTION 8 — WRITE OUTPUTS

cat("\n[8] Writing CSV outputs to:", OUTPUT_DIR, "\n")

fwrite(tp[, .(datetime_utc, log_RV_hat, sigma2_t)],
       file.path(OUTPUT_DIR, "mcgarch_forecasts.csv"),
       dateTimeAs = "write.csv")

fwrite(wfv_df, file.path(OUTPUT_DIR, "mcgarch_wfv.csv"))

metadata <- data.table(
  phi     = fit_final$params$phi,
  omega = fit_final$params$omega,
  alpha   = fit_final$params$alpha,
  beta   = fit_final$params$beta,
  persistence = fit_final$params$persistence,
  resid_var = fit_final$params$resid_var,
  n_obs_fit = nrow(rv_all[!is.na(log_RV)]),
  data_start  = as.character(DATA_START),
  data_end = as.character(DATA_END),
  fit_date   = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)
fwrite(metadata, file.path(OUTPUT_DIR, "mcgarch_metadata.csv"))

cat(sprintf("  mcgarch_btc_forecasts.csv : %d rows  (datetime_utc, log_RV_hat, sigma2_t)\n",
            nrow(tp)))
cat(sprintf("  mcgarch_btc_wfv.csv  : %d folds\n", nrow(wfv_df)))
cat("  mcgarch_btc_metadata.csv  : 1 row  (params + data_start/end + fit_date)\n")