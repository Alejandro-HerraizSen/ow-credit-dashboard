"""
Credit Decisioning Model — Core Logic
======================================
WoE-IV Credit Scorecard grounded in information theory / KL divergence.

Mathematical framework
----------------------
  WoE_i = ln( P(X=i | Y=1) / P(X=i | Y=0) )
         = ln( (Events_i / N_events) / (NonEvents_i / N_non_events) )

  IV = Σ_i [ P(X=i|Y=1) - P(X=i|Y=0) ] × WoE_i
     = KL(events ‖ non-events) + KL(non-events ‖ events)   [symmetric KL divergence]

  Logistic regression on WoE-transformed features is the MLE estimator of the
  latent creditworthiness model:
      Z_i ~ N(Xβ, 1),  default ⟺ Z_i < 0  →  P(default|X) = σ(Xβ)

Score scaling (PDO — Points to Double Odds)
-------------------------------------------
  Factor = PDO / ln(2)
  Offset = BaseScore − Factor × ln(BaseOdds)
  Score  = Offset − Factor × log_odds          (higher score = lower risk)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
CONTINUOUS_COLS = ["income", "loan_amount", "term_length", "install_to_inc", "schufa", "num_applic"]
CATEGORICAL_COLS = ["occup", "marital"]
ALL_FEATURES = CONTINUOUS_COLS + CATEGORICAL_COLS
TARGET = "target_var"

N_BINS = 10       # Equal-frequency bins for continuous features
EPSILON = 0.5     # Laplace smoothing for zero-event bins (prevents ln(0))
PDO = 20          # Points to double odds
BASE_ODDS = 4.0   # Expected non-default : default ≈ 80:20 = 4:1
BASE_SCORE = 600  # Score anchor

OCCUP_CODE_MAP = {"1": "Worker", "2": "Employee", "3": "Student"}

IV_THRESHOLDS = [
    (0.02,        "Useless"),
    (0.10,        "Weak"),
    (0.30,        "Medium"),
    (0.50,        "Strong"),
    (float("inf"), "Suspicious (overfit risk)"),
]


# ── DataCleaner ────────────────────────────────────────────────────────────────
class DataCleaner:
    """Standardise raw CSV data: missing values, occupation codes, dtypes."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Unify both missing-value representations
        for col in df.columns:
            df[col] = df[col].replace({"": np.nan, "Not avail.": np.nan})

        # Recode numeric occupation codes before any encoding step
        if "occup" in df.columns:
            df["occup"] = (
                df["occup"]
                .astype(str)
                .map(lambda v: OCCUP_CODE_MAP.get(v, v))
                .replace("nan", np.nan)
            )

        # Cast continuous columns to float
        for col in CONTINUOUS_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Extract observation year from OBS_DATE ("18JUL2018 - 00:00:00")
        if "OBS_DATE" in df.columns:
            def _parse_year(x):
                if pd.isna(x) or str(x).strip() in ("", "nan"):
                    return np.nan
                try:
                    return pd.to_datetime(str(x).split(" ")[0], format="%d%b%Y").year
                except Exception:
                    return np.nan
            df["obs_year"] = df["OBS_DATE"].apply(_parse_year)

        # Cast target to numeric
        if TARGET in df.columns:
            df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

        return df

    def get_cleaning_report(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
        raw_replaced = df_raw.replace({"": np.nan, "Not avail.": np.nan})
        feat_cols = [c for c in ALL_FEATURES if c in df_raw.columns]
        occup_recoded = int(
            df_raw["occup"].astype(str).isin(["1", "2", "3"]).sum()
        ) if "occup" in df_raw.columns else 0
        return {
            "rows_before": len(df_raw),
            "rows_after": int(df_clean[TARGET].notna().sum()),
            "missing_before": raw_replaced[feat_cols].isnull().sum().to_dict(),
            "missing_after": df_clean[[c for c in feat_cols if c in df_clean.columns]].isnull().sum().to_dict(),
            "occup_recoded": occup_recoded,
            "total_missing_before": int(raw_replaced[feat_cols].isnull().sum().sum()),
            "total_missing_after": int(df_clean[[c for c in feat_cols if c in df_clean.columns]].isnull().sum().sum()),
        }


# ── WoEEncoder ─────────────────────────────────────────────────────────────────
class WoEEncoder:
    """
    Weight of Evidence encoder — built from scratch, no external scorecard packages.

    Continuous features: equal-frequency binning (pd.qcut), missing → own bin.
    Categorical features: WoE per category, missing → own bin.
    Zero-event bins: Laplace smoothing (ε = 0.5) to prevent ln(0).

    CRITICAL: must be fitted on training data only to avoid target leakage.
    """

    def __init__(self, n_bins: int = N_BINS, epsilon: float = EPSILON):
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.bins_map: dict = {}       # feature → ("continuous", edges) | ("categorical", None)
        self.woe_map: dict = {}        # feature → {bin_str: woe_value}
        self.iv_map: dict = {}         # feature → IV float
        self.bin_stats: dict = {}      # feature → DataFrame with per-bin stats
        self.smoothed_bins: dict = {}  # feature → list of bin labels that needed smoothing

    # ── internals ─────────────────────────────────────────────────────────────

    def _compute_woe_stats(
        self, series_binned: pd.Series, y: pd.Series
    ) -> tuple:
        """Core WoE / IV computation from a pre-binned series."""
        df = pd.DataFrame({"bin": series_binned.values, "y": y.values})
        N_ev = float(y.sum())
        N_nev = float(len(y) - N_ev)

        rows, smoothed = [], []

        # Non-missing bins in sorted string order
        unique = sorted(df["bin"][~pd.isna(df["bin"])].unique(), key=str)
        for bl in unique:
            mask = df["bin"] == bl
            n = int(mask.sum())
            raw_ev = float(df.loc[mask, "y"].sum())
            raw_nev = float(n - raw_ev)
            er = raw_ev / n if n else 0.0

            needs_smooth = raw_ev == 0 or raw_nev == 0
            ev = raw_ev + (self.epsilon if needs_smooth else 0)
            nev = raw_nev + (self.epsilon if needs_smooth else 0)
            if needs_smooth:
                smoothed.append(str(bl))

            d_ev = ev / N_ev
            d_nev = nev / N_nev
            woe = float(np.log(d_ev / d_nev))
            iv_c = float((d_ev - d_nev) * woe)

            rows.append(dict(
                bin=str(bl), count=n,
                events=int(raw_ev), non_events=int(raw_nev),
                event_rate=er, woe=woe, iv_contribution=iv_c,
                smoothed=needs_smooth,
            ))

        # Missing bin
        miss = pd.isna(df["bin"])
        if miss.sum() > 0:
            n = int(miss.sum())
            raw_ev = float(df.loc[miss, "y"].sum())
            raw_nev = float(n - raw_ev)
            er = raw_ev / n if n else 0.0
            needs_smooth = raw_ev == 0 or raw_nev == 0
            ev = raw_ev + (self.epsilon if needs_smooth else 0)
            nev = raw_nev + (self.epsilon if needs_smooth else 0)
            if needs_smooth:
                smoothed.append("Missing")
            d_ev = ev / N_ev
            d_nev = nev / N_nev
            woe = float(np.log(d_ev / d_nev))
            iv_c = float((d_ev - d_nev) * woe)
            rows.append(dict(
                bin="Missing", count=n,
                events=int(raw_ev), non_events=int(raw_nev),
                event_rate=er, woe=woe, iv_contribution=iv_c,
                smoothed=needs_smooth,
            ))

        stats_df = pd.DataFrame(rows)
        total_iv = float(stats_df["iv_contribution"].sum())
        return stats_df, total_iv, smoothed

    def _bin_continuous(self, series: pd.Series) -> tuple:
        """Equal-frequency binning with graceful fallback for low-cardinality features."""
        non_null = series.dropna()
        edges = None
        for q in [self.n_bins, 5, 3, 2]:
            try:
                _, edges = pd.qcut(non_null, q=q, retbins=True, duplicates="drop")
                if len(edges) > 2:
                    break
            except Exception:
                continue

        if edges is None or len(edges) <= 2:
            med = float(non_null.median())
            edges = np.array([-np.inf, med, np.inf])
        else:
            edges = edges.copy()
            edges[0] = -np.inf
            edges[-1] = np.inf

        binned = pd.cut(series, bins=edges, include_lowest=True)
        return binned, edges

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        """Fit on training data ONLY — must not see test labels."""
        y = y.astype(float).reset_index(drop=True)

        for col in CONTINUOUS_COLS:
            if col not in X.columns:
                continue
            s = X[col].reset_index(drop=True)
            binned, edges = self._bin_continuous(s)
            self.bins_map[col] = ("continuous", edges)
            stats_df, iv, smoothed = self._compute_woe_stats(binned, y)
            self.woe_map[col] = dict(zip(stats_df["bin"], stats_df["woe"]))
            self.iv_map[col] = iv
            self.bin_stats[col] = stats_df
            self.smoothed_bins[col] = smoothed

        for col in CATEGORICAL_COLS:
            if col not in X.columns:
                continue
            s = X[col].reset_index(drop=True)
            self.bins_map[col] = ("categorical", None)
            stats_df, iv, smoothed = self._compute_woe_stats(s, y)
            self.woe_map[col] = dict(zip(stats_df["bin"], stats_df["woe"]))
            self.iv_map[col] = iv
            self.bin_stats[col] = stats_df
            self.smoothed_bins[col] = smoothed

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map features to WoE values. Unknown bins → WoE = 0 (neutral)."""
        X_woe = pd.DataFrame(index=X.index)

        for col in CONTINUOUS_COLS + CATEGORICAL_COLS:
            if col not in X.columns or col not in self.bins_map:
                continue
            kind, edges = self.bins_map[col]

            if kind == "continuous":
                binned = pd.cut(X[col], bins=edges, include_lowest=True)
                bin_str = binned.astype(str).where(X[col].notna(), "Missing")
            else:
                bin_str = X[col].astype(str).where(X[col].notna(), "Missing")

            X_woe[col + "_woe"] = bin_str.map(self.woe_map[col]).fillna(0.0)

        return X_woe

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_iv_table(self) -> pd.DataFrame:
        rows = []
        for col, iv in self.iv_map.items():
            label = next(lbl for thresh, lbl in IV_THRESHOLDS if iv < thresh)
            rows.append({"Feature": col, "IV": round(iv, 4), "Predictive Power": label})
        return (
            pd.DataFrame(rows)
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )

    def monotonic_check(self, feature: str) -> tuple:
        """
        Returns (is_monotonic: bool, direction: str).
        Non-monotonic WoE in continuous bins is a red flag for overfitting /
        insufficient data per bin.
        """
        if (
            feature not in self.bin_stats
            or self.bins_map.get(feature, (None,))[0] != "continuous"
        ):
            return True, "n/a"
        ordered = self.bin_stats[feature][self.bin_stats[feature]["bin"] != "Missing"]
        woe_vals = ordered["woe"].values
        if len(woe_vals) <= 2:
            return True, "monotonic"
        diffs = np.diff(woe_vals)
        if np.all(diffs >= -1e-8):
            return True, "increasing ↑"
        if np.all(diffs <= 1e-8):
            return True, "decreasing ↓"
        return False, "non-monotonic ⚠"


# ── ScorecardModel ─────────────────────────────────────────────────────────────
class ScorecardModel:
    """
    WoE-encoded logistic regression with PDO score scaling and probability calibration.

    Two sub-models are fitted:
      • self.lr  — raw LR on full training set, used for coefficient extraction
                   and score calculation (interpretable weights).
      • self.calibrated — CalibratedClassifierCV(sigmoid), corrects the
                   probability distortion caused by class_weight='balanced'.
    """

    def __init__(self, pdo=PDO, base_odds=BASE_ODDS, base_score=BASE_SCORE):
        self.pdo = pdo
        self.base_odds = base_odds
        self.base_score = base_score
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        self.woe_encoder = WoEEncoder()
        self.lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        self.calibrated = None
        self.feature_names_: list = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ScorecardModel":
        # WoE fitted on training data only — zero leakage
        X_woe = self.woe_encoder.fit_transform(X_train, y_train)
        self.feature_names_ = X_woe.columns.tolist()

        # Raw LR for interpretable coefficients
        self.lr.fit(X_woe, y_train)

        # Calibrated model for realistic PD (class_weight distorts raw probs)
        cal_base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        self.calibrated = CalibratedClassifierCV(cal_base, method="sigmoid", cv=5)
        self.calibrated.fit(X_woe, y_train)
        return self

    def _to_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.woe_encoder.transform(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default."""
        return self.calibrated.predict_proba(self._to_woe(X))[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Integer credit score 0–1000 (higher = lower risk)."""
        X_woe = self._to_woe(X)
        log_odds = self.lr.intercept_[0] + (X_woe.values @ self.lr.coef_[0])
        return np.clip(np.round(self.offset - self.factor * log_odds).astype(int), 0, 1000)

    def get_feature_contributions(self, X_row: pd.DataFrame) -> tuple:
        """
        Per-feature score contributions for a single applicant.
        contribution_i = -Factor × β_i × WoE_i(x_i)

        Returns (contributions_df, base_score_component).
        """
        X_woe = self._to_woe(X_row)
        coefs = self.lr.coef_[0]
        base = self.offset - self.factor * self.lr.intercept_[0]

        rows = []
        for i, fname in enumerate(self.feature_names_):
            woe_val = float(X_woe[fname].iloc[0])
            coef = float(coefs[i])
            contrib = -self.factor * coef * woe_val
            rows.append({
                "Feature": fname.replace("_woe", ""),
                "WoE": round(woe_val, 4),
                "Coefficient (β)": round(coef, 4),
                "Score Points": round(contrib, 1),
            })
        return pd.DataFrame(rows), round(base, 1)

    def get_scorecard_table(self) -> pd.DataFrame:
        """Full scorecard: every feature × bin mapped to score points."""
        coef_map = dict(zip(self.feature_names_, self.lr.coef_[0]))
        rows = []
        for col, stats_df in self.woe_encoder.bin_stats.items():
            coef = coef_map.get(col + "_woe", 0.0)
            for _, row in stats_df.iterrows():
                rows.append({
                    "Feature": col,
                    "Bin": row["bin"],
                    "Count": row["count"],
                    "Event Rate": row["event_rate"],
                    "WoE": round(row["woe"], 4),
                    "β (LR coef)": round(coef, 4),
                    "Score Points": round(-self.factor * coef * row["woe"], 1),
                    "Smoothed": row["smoothed"],
                })
        return pd.DataFrame(rows)


# ── BaselineModels ─────────────────────────────────────────────────────────────
class BaselineModels:
    """
    Three comparison pipelines using standard sklearn preprocessing:
      - median imputation for numerics
      - constant-fill + ordinal encoding for categoricals

    No WoE transformation — intentionally naive to highlight scorecard advantage.
    """

    MODEL_NOTES = {
        "Vanilla LR": (
            "Logistic regression with the same features as the scorecard, but using "
            "median imputation + ordinal encoding instead of WoE transformation. "
            "Directly comparable to the scorecard (same algorithm, different preprocessing). "
            "Shows how much WoE feature engineering adds over raw features."
        ),
        "Decision Tree": (
            "Decision tree (max_depth=5) with median imputation + ordinal encoding. "
            "Non-linear and interpretable via tree structure, but prone to instability "
            "and limited capacity at depth 5."
        ),
        "Random Forest": (
            "Ensemble of 100 decision trees with bagging. More powerful than a single tree, "
            "but opaque — no individual feature-weight interpretability. "
            "Strong baseline for tabular data."
        ),
        "Gradient Boosting": (
            "HistGradientBoostingClassifier (sklearn's XGBoost-equivalent). Sequentially fits "
            "shallow trees on residuals: F_m(x) = F_{m-1}(x) + η·h_m(x), where h_m minimises "
            "the negative log-likelihood gradient. Handles missing values natively — no imputation "
            "needed for numeric features. Often the strongest off-the-shelf model on tabular data."
        ),
        "SVM (RBF kernel)": (
            "Support Vector Machine with radial basis function kernel: "
            "K(x,x') = exp(−γ‖x−x'‖²). Finds the maximum-margin hyperplane in a high-dimensional "
            "feature space. Requires feature scaling (StandardScaler) and cannot handle NaN — "
            "median imputation applied. Probability estimates via Platt scaling (5-fold CV). "
            "Computationally expensive on large datasets; scales as O(n² – n³)."
        ),
    }

    def __init__(self):
        self.pipelines: dict = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaselineModels":
        num_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

        # Standard preprocessor: median imputation + ordinal encoding
        std_prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_cols),
        ], remainder="drop")

        # HistGBT preprocessor: numeric NaN handled natively, only encode categoricals
        hgbt_prep = ColumnTransformer([
            ("num", "passthrough", num_cols),   # NaN kept — HistGBT handles it
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_cols),
        ], remainder="drop")

        pipelines_cfg = {
            "Vanilla LR": (
                std_prep,
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                False,  # no scaler
            ),
            "Decision Tree": (
                std_prep,
                DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
                False,
            ),
            "Random Forest": (
                std_prep,
                RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
                False,
            ),
            "Gradient Boosting": (
                hgbt_prep,
                HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, max_depth=4,
                    class_weight="balanced", random_state=42,
                ),
                False,
            ),
            "SVM (RBF kernel)": (
                std_prep,
                SVC(kernel="rbf", probability=True, class_weight="balanced",
                    C=1.0, gamma="scale", random_state=42),
                True,  # needs StandardScaler
            ),
        }

        for name, (prep, clf, needs_scale) in pipelines_cfg.items():
            steps = [("prep", prep)]
            if needs_scale:
                steps.append(("scaler", StandardScaler()))
            steps.append(("clf", clf))
            pipe = Pipeline(steps)
            pipe.fit(X_train, y_train)
            self.pipelines[name] = pipe

        return self

    def predict_proba_all(self, X: pd.DataFrame) -> dict:
        return {name: pipe.predict_proba(X)[:, 1] for name, pipe in self.pipelines.items()}


# ── Metric functions ───────────────────────────────────────────────────────────

def compute_auc(y_true, y_prob) -> float:
    return float(roc_auc_score(y_true, y_prob))

def compute_gini(y_true, y_prob) -> float:
    """Gini = 2·AUC − 1.  Normalised area between ROC curve and the diagonal."""
    return float(2 * compute_auc(y_true, y_prob) - 1)

def compute_ks(y_true, y_prob) -> float:
    """KS = max_t |F_events(t) − F_non_events(t)|  via two-sample KS test."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    stat, _ = ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])
    return float(stat)

def compute_brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))

def compute_roc_curve_data(y_true, y_prob):
    return roc_curve(y_true, y_prob)

def compute_ks_curve_data(y_true, y_prob) -> pd.DataFrame:
    """Cumulative event / non-event distributions sorted by descending score."""
    df = pd.DataFrame({"y": np.asarray(y_true), "prob": np.asarray(y_prob)})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    tot_ev = (df["y"] == 1).sum()
    tot_nev = (df["y"] == 0).sum()
    df["cum_events"] = (df["y"] == 1).cumsum() / tot_ev
    df["cum_non_events"] = (df["y"] == 0).cumsum() / tot_nev
    df["percentile"] = (df.index + 1) / len(df)
    df["ks_diff"] = (df["cum_events"] - df["cum_non_events"]).abs()
    return df


# ── End-to-end pipeline ────────────────────────────────────────────────────────

def run_full_pipeline(df_raw: pd.DataFrame) -> dict:
    """
    1. Clean  →  2. Stratified 80/20 split  →  3. Fit WoE on X_train ONLY
    →  4. Fit LR + calibrate  →  5. Fit baselines  →  6. Evaluate on X_test
    """
    cleaner = DataCleaner()
    df_clean = cleaner.transform(df_raw)
    cleaning_report = cleaner.get_cleaning_report(df_raw, df_clean)

    df_model = df_clean.dropna(subset=[TARGET]).copy()
    df_model[TARGET] = df_model[TARGET].astype(int)

    X = df_model[ALL_FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scorecard = ScorecardModel()
    scorecard.fit(X_train, y_train)

    baselines = BaselineModels()
    baselines.fit(X_train, y_train)

    sc_proba = scorecard.predict_proba(X_test)
    bl_probas = baselines.predict_proba_all(X_test)

    def _metrics(proba):
        return {
            "AUC":   round(compute_auc(y_test, proba), 4),
            "Gini":  round(compute_gini(y_test, proba), 4),
            "KS":    round(compute_ks(y_test, proba), 4),
            "Brier": round(compute_brier(y_test, proba), 4),
        }

    metrics = {"WoE Scorecard": _metrics(sc_proba)}
    for name, proba in bl_probas.items():
        metrics[name] = _metrics(proba)

    return {
        "cleaner":             cleaner,
        "cleaning_report":     cleaning_report,
        "scorecard":           scorecard,
        "baselines":           baselines,
        "metrics":             metrics,
        "X_train":             X_train,
        "X_test":              X_test,
        "y_train":             y_train,
        "y_test":              y_test,
        "df_clean":            df_clean,
        "df_model":            df_model,
        "sc_proba_test":       sc_proba,
        "baseline_probas_test": bl_probas,
    }
