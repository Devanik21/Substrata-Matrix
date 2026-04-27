"""
preprocessing.py — ClusterX Advanced Data Preprocessing Engine
================================================================
Handles: loading, profiling, imputation, outlier detection,
         scaling, encoding, feature selection, and pipeline assembly.
Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import io
import time
import warnings
import logging
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, MaxAbsScaler,
    LabelEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────

class ScalerType(str, Enum):
    STANDARD       = "standard"
    MINMAX         = "minmax"
    ROBUST         = "robust"
    QUANTILE_NORM  = "quantile_normal"
    QUANTILE_UNIF  = "quantile_uniform"
    POWER_YEO      = "power_yeo"
    POWER_BOX      = "power_box"
    MAXABS         = "maxabs"
    NONE           = "none"


class ImputeStrategy(str, Enum):
    MEAN       = "mean"
    MEDIAN     = "median"
    MODE       = "most_frequent"
    CONSTANT   = "constant"
    KNN        = "knn"
    ITERATIVE  = "iterative"
    DROP       = "drop"


class OutlierMethod(str, Enum):
    ZSCORE         = "zscore"
    IQR            = "iqr"
    ISOLATION      = "isolation_forest"
    LOF            = "lof"
    ELLIPTIC       = "elliptic_envelope"
    NONE           = "none"


class OutlierAction(str, Enum):
    REMOVE  = "remove"
    CLIP    = "clip"
    FLAG    = "flag"
    NONE    = "none"


class FeatureSelectionMethod(str, Enum):
    VARIANCE      = "variance_threshold"
    CORRELATION   = "correlation"
    PCA_REDUCE    = "pca_reduce"
    MANUAL        = "manual"
    NONE          = "none"


# ──────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    missing_pct: float
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    q25: Optional[float]
    median: Optional[float]
    q75: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    is_numeric: bool
    is_categorical: bool
    is_binary: bool
    normality_p: Optional[float]
    outlier_count: int
    outlier_pct: float
    top_values: Optional[Dict[str, int]]


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_binary: int
    total_missing: int
    total_missing_pct: float
    memory_mb: float
    duplicate_rows: int
    duplicate_pct: float
    column_profiles: Dict[str, ColumnProfile]
    correlation_matrix: Optional[pd.DataFrame]
    high_corr_pairs: List[Tuple[str, str, float]]
    data_hash: str
    warnings: List[str]


@dataclass
class PreprocessingConfig:
    numeric_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    drop_columns: List[str] = field(default_factory=list)
    impute_strategy: ImputeStrategy = ImputeStrategy.MEDIAN
    impute_constant: float = 0.0
    knn_neighbors: int = 5
    scaler_type: ScalerType = ScalerType.STANDARD
    outlier_method: OutlierMethod = OutlierMethod.NONE
    outlier_action: OutlierAction = OutlierAction.FLAG
    outlier_threshold: float = 3.0
    outlier_contamination: float = 0.05
    feature_selection: FeatureSelectionMethod = FeatureSelectionMethod.NONE
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    pca_n_components: Optional[int] = None
    pca_variance_explained: float = 0.95
    encode_categoricals: bool = True
    max_categorical_cardinality: int = 50
    random_state: int = 42


@dataclass
class PreprocessingResult:
    X_original: pd.DataFrame
    X_processed: pd.DataFrame
    feature_names: List[str]
    dropped_columns: List[str]
    outlier_mask: Optional[np.ndarray]
    n_outliers: int
    scaler: Optional[Any]
    imputer: Optional[Any]
    feature_selector: Optional[Any]
    config: PreprocessingConfig
    profile: DataProfile
    log: List[str]
    warnings: List[str]


# ──────────────────────────────────────────────────────────────────
# DATA LOADER
# ──────────────────────────────────────────────────────────────────

class DataLoader:
    """Loads data from CSV, Excel, JSON, and parquet with auto-detection."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"}
    MAX_ROWS = 500_000
    MAX_COLS = 2000

    def __init__(self, max_rows: int = MAX_ROWS, max_cols: int = MAX_COLS):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self._load_log: List[str] = []

    def load(self, source: Union[str, io.BytesIO, io.StringIO],
             filename: str = "", **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        self._load_log = []
        ext = self._detect_extension(filename)
        try:
            if ext in (".csv", ".tsv"):
                sep = "\t" if ext == ".tsv" else ","
                df = self._load_csv(source, sep=sep, **kwargs)
            elif ext in (".xlsx", ".xls"):
                df = self._load_excel(source, **kwargs)
            elif ext == ".json":
                df = self._load_json(source, **kwargs)
            elif ext == ".parquet":
                df = pd.read_parquet(source)
            else:
                # Try CSV as default
                df = self._load_csv(source, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}") from e

        df = self._sanitize(df)
        self._log(f"Loaded {len(df)} rows × {len(df.columns)} columns")
        return df, self._load_log

    def _load_csv(self, source, sep=",", **kwargs):
        parse_kwargs = dict(
            sep=sep, low_memory=False,
            na_values=["NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null", "None", "none", "", " "],
            keep_default_na=True,
        )
        parse_kwargs.update(kwargs)
        df = pd.read_csv(source, **parse_kwargs)
        if len(df) > self.max_rows:
            self._log(f"WARNING: Truncated to {self.max_rows} rows (was {len(df)})")
            df = df.sample(self.max_rows, random_state=42).reset_index(drop=True)
        return df

    def _load_excel(self, source, **kwargs):
        sheets = pd.read_excel(source, sheet_name=None, **kwargs)
        if len(sheets) > 1:
            self._log(f"Multiple sheets detected: {list(sheets.keys())}. Using first.")
        df = list(sheets.values())[0]
        return df

    def _load_json(self, source, **kwargs):
        try:
            df = pd.read_json(source, **kwargs)
        except ValueError:
            df = pd.read_json(source, orient="records", **kwargs)
        return df

    def _sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean column names
        df.columns = [str(c).strip().replace(" ", "_").replace(".", "_")
                      .replace("/", "_").replace("\\", "_")
                      .replace("(", "").replace(")", "") for c in df.columns]
        # Deduplicate columns
        seen = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
        # Truncate columns
        if len(df.columns) > self.max_cols:
            df = df.iloc[:, :self.max_cols]
            self._log(f"WARNING: Truncated to {self.max_cols} columns")
        return df

    def _detect_extension(self, filename: str) -> str:
        if "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ".csv"

    def _log(self, msg: str):
        self._load_log.append(msg)
        logger.info(msg)


# ──────────────────────────────────────────────────────────────────
# DATA PROFILER
# ──────────────────────────────────────────────────────────────────

class DataProfiler:
    """Deep statistical profiling of a DataFrame."""

    def __init__(self, compute_correlation: bool = True,
                 normality_sample: int = 5000):
        self.compute_correlation = compute_correlation
        self.normality_sample = normality_sample

    def profile(self, df: pd.DataFrame) -> DataProfile:
        warnings_list = []
        col_profiles = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in df.columns:
            try:
                cp = self._profile_column(df[col], col)
                col_profiles[col] = cp
            except Exception as e:
                warnings_list.append(f"Could not profile column {col}: {e}")

        # Correlation matrix
        corr_matrix = None
        high_corr_pairs = []
        if self.compute_correlation and len(numeric_cols) >= 2:
            try:
                num_df = df[numeric_cols].copy()
                corr_matrix = num_df.corr(method="pearson")
                high_corr_pairs = self._find_high_correlations(corr_matrix)
            except Exception:
                pass

        # Duplicate rows
        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / max(len(df), 1) * 100, 2)

        total_missing = int(df.isnull().sum().sum())
        total_cells = len(df) * len(df.columns)

        # Data hash for caching
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df.head(1000)).values.tobytes()
        ).hexdigest()[:16]

        return DataProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            n_numeric=len(numeric_cols),
            n_categorical=len(cat_cols),
            n_binary=sum(1 for cp in col_profiles.values() if cp.is_binary),
            total_missing=total_missing,
            total_missing_pct=round(total_missing / max(total_cells, 1) * 100, 2),
            memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 3),
            duplicate_rows=dup_count,
            duplicate_pct=dup_pct,
            column_profiles=col_profiles,
            correlation_matrix=corr_matrix,
            high_corr_pairs=high_corr_pairs,
            data_hash=data_hash,
            warnings=warnings_list,
        )

    def _profile_column(self, series: pd.Series, name: str) -> ColumnProfile:
        is_numeric = pd.api.types.is_numeric_dtype(series)
        n_unique = int(series.nunique())
        n_missing = int(series.isnull().sum())
        missing_pct = round(n_missing / max(len(series), 1) * 100, 2)
        is_binary = n_unique == 2
        is_cat = not is_numeric or n_unique <= 20

        mean = std = mn = mx = q25 = med = q75 = skew = kurt = norm_p = None
        outlier_count = 0
        outlier_pct = 0.0
        top_values = None

        if is_numeric:
            clean = series.dropna()
            if len(clean) > 0:
                mean = float(clean.mean())
                std = float(clean.std())
                mn = float(clean.min())
                mx = float(clean.max())
                q25 = float(clean.quantile(0.25))
                med = float(clean.median())
                q75 = float(clean.quantile(0.75))
                skew = float(clean.skew())
                kurt = float(clean.kurtosis())
                # Outlier count via IQR
                iqr = q75 - q25
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                outlier_count = int(((clean < lower) | (clean > upper)).sum())
                outlier_pct = round(outlier_count / max(len(clean), 1) * 100, 2)
                # Normality test
                sample = clean.sample(min(self.normality_sample, len(clean)),
                                      random_state=42)
                if len(sample) >= 8:
                    try:
                        _, norm_p = normaltest(sample)
                        norm_p = float(norm_p)
                    except Exception:
                        norm_p = None
        else:
            top_values = dict(series.value_counts().head(10))

        return ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            n_unique=n_unique,
            n_missing=n_missing,
            missing_pct=missing_pct,
            mean=mean, std=std, min=mn, max=mx,
            q25=q25, median=med, q75=q75,
            skewness=skew, kurtosis=kurt,
            is_numeric=is_numeric,
            is_categorical=is_cat,
            is_binary=is_binary,
            normality_p=norm_p,
            outlier_count=outlier_count,
            outlier_pct=outlier_pct,
            top_values=top_values,
        )

    def _find_high_correlations(self, corr: pd.DataFrame,
                                threshold: float = 0.90) -> List[Tuple[str, str, float]]:
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = abs(corr.iloc[i, j])
                if val >= threshold:
                    pairs.append((cols[i], cols[j], round(float(val), 4)))
        return sorted(pairs, key=lambda x: -x[2])


# ──────────────────────────────────────────────────────────────────
# MISSING VALUE HANDLER
# ──────────────────────────────────────────────────────────────────

class MissingValueHandler:
    """Multiple strategies for handling missing values."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._imputer = None
        self._fit_log: List[str] = []

    def fit_transform(self, df: pd.DataFrame,
                      numeric_cols: List[str]) -> Tuple[pd.DataFrame, Any]:
        strategy = self.config.impute_strategy
        self._fit_log = []

        if strategy == ImputeStrategy.DROP:
            n_before = len(df)
            df = df.dropna(subset=numeric_cols).reset_index(drop=True)
            n_after = len(df)
            self._fit_log.append(f"Dropped {n_before - n_after} rows with missing values")
            return df, None

        if not df[numeric_cols].isnull().any().any():
            self._fit_log.append("No missing values in numeric columns — skipping imputation")
            return df, None

        if strategy == ImputeStrategy.KNN:
            imputer = KNNImputer(
                n_neighbors=self.config.knn_neighbors,
                weights="distance"
            )
        elif strategy == ImputeStrategy.CONSTANT:
            imputer = SimpleImputer(
                strategy="constant",
                fill_value=self.config.impute_constant
            )
        elif strategy == ImputeStrategy.ITERATIVE:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(
                    max_iter=10,
                    random_state=self.config.random_state,
                    n_nearest_features=min(10, len(numeric_cols))
                )
            except ImportError:
                imputer = SimpleImputer(strategy="median")
                self._fit_log.append("IterativeImputer unavailable; falling back to median")
        else:
            imputer = SimpleImputer(strategy=strategy.value)

        n_missing_before = int(df[numeric_cols].isnull().sum().sum())
        arr = imputer.fit_transform(df[numeric_cols])
        df = df.copy()
        df[numeric_cols] = arr
        self._fit_log.append(
            f"Imputed {n_missing_before} missing values using {strategy.value}"
        )
        self._imputer = imputer
        return df, imputer

    def transform(self, df: pd.DataFrame,
                  numeric_cols: List[str],
                  imputer: Any) -> pd.DataFrame:
        if imputer is None:
            return df
        arr = imputer.transform(df[numeric_cols])
        df = df.copy()
        df[numeric_cols] = arr
        return df

    @property
    def log(self) -> List[str]:
        return self._fit_log


# ──────────────────────────────────────────────────────────────────
# OUTLIER DETECTOR
# ──────────────────────────────────────────────────────────────────

class OutlierDetector:
    """Detects and handles outliers using multiple methods."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._model = None
        self._log: List[str] = []

    def detect(self, X: np.ndarray) -> np.ndarray:
        """Returns boolean mask where True = outlier."""
        method = self.config.outlier_method
        self._log = []

        if method == OutlierMethod.NONE:
            return np.zeros(len(X), dtype=bool)

        try:
            if method == OutlierMethod.ZSCORE:
                mask = self._zscore_detection(X)
            elif method == OutlierMethod.IQR:
                mask = self._iqr_detection(X)
            elif method == OutlierMethod.ISOLATION:
                mask = self._isolation_forest_detection(X)
            elif method == OutlierMethod.LOF:
                mask = self._lof_detection(X)
            elif method == OutlierMethod.ELLIPTIC:
                mask = self._elliptic_envelope_detection(X)
            else:
                mask = np.zeros(len(X), dtype=bool)
        except Exception as e:
            self._log.append(f"Outlier detection failed: {e}; returning no outliers")
            mask = np.zeros(len(X), dtype=bool)

        n_outliers = int(mask.sum())
        self._log.append(
            f"Detected {n_outliers} outliers ({n_outliers/len(X)*100:.1f}%) "
            f"using {method.value}"
        )
        return mask

    def _zscore_detection(self, X: np.ndarray) -> np.ndarray:
        z = np.abs(stats.zscore(X, nan_policy="omit"))
        return (z > self.config.outlier_threshold).any(axis=1)

    def _iqr_detection(self, X: np.ndarray) -> np.ndarray:
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = ((X < lower) | (X > upper)).any(axis=1)
        return mask

    def _isolation_forest_detection(self, X: np.ndarray) -> np.ndarray:
        model = IsolationForest(
            contamination=self.config.outlier_contamination,
            random_state=self.config.random_state,
            n_estimators=100,
        )
        preds = model.fit_predict(X)
        self._model = model
        return preds == -1

    def _lof_detection(self, X: np.ndarray) -> np.ndarray:
        n = min(20, len(X) // 5)
        n = max(n, 3)
        model = LocalOutlierFactor(
            n_neighbors=n,
            contamination=self.config.outlier_contamination,
        )
        preds = model.fit_predict(X)
        return preds == -1

    def _elliptic_envelope_detection(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] > X.shape[0]:
            self._log.append("Too many features for EllipticEnvelope; using IQR fallback")
            return self._iqr_detection(X)
        model = EllipticEnvelope(
            contamination=self.config.outlier_contamination,
            random_state=self.config.random_state,
        )
        preds = model.fit_predict(X)
        self._model = model
        return preds == -1

    def apply_action(self, df: pd.DataFrame,
                     mask: np.ndarray,
                     numeric_cols: List[str]) -> pd.DataFrame:
        action = self.config.outlier_action
        df = df.copy()
        if action == OutlierAction.REMOVE:
            df = df[~mask].reset_index(drop=True)
            self._log.append(f"Removed {mask.sum()} outlier rows")
        elif action == OutlierAction.CLIP:
            for col in numeric_cols:
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q01, upper=q99)
            self._log.append(f"Clipped {mask.sum()} outlier rows to [1%, 99%] range")
        elif action == OutlierAction.FLAG:
            df["__outlier__"] = mask.astype(int)
            self._log.append(f"Flagged {mask.sum()} outliers with __outlier__ column")
        return df

    @property
    def log(self) -> List[str]:
        return self._log


# ──────────────────────────────────────────────────────────────────
# FEATURE SCALER
# ──────────────────────────────────────────────────────────────────

class FeatureScaler:
    """Wraps sklearn scalers with friendly interface."""

    SCALER_MAP = {
        ScalerType.STANDARD:      lambda: StandardScaler(),
        ScalerType.MINMAX:        lambda: MinMaxScaler(),
        ScalerType.ROBUST:        lambda: RobustScaler(quantile_range=(10, 90)),
        ScalerType.QUANTILE_NORM: lambda: QuantileTransformer(output_distribution="normal",
                                                               n_quantiles=1000),
        ScalerType.QUANTILE_UNIF: lambda: QuantileTransformer(output_distribution="uniform",
                                                               n_quantiles=1000),
        ScalerType.POWER_YEO:     lambda: PowerTransformer(method="yeo-johnson"),
        ScalerType.POWER_BOX:     lambda: PowerTransformer(method="box-cox"),
        ScalerType.MAXABS:        lambda: MaxAbsScaler(),
        ScalerType.NONE:          lambda: None,
    }

    def __init__(self, scaler_type: ScalerType = ScalerType.STANDARD):
        self.scaler_type = scaler_type
        self._scaler = None

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[Any]]:
        if self.scaler_type == ScalerType.NONE:
            return X, None
        factory = self.SCALER_MAP.get(self.scaler_type)
        if factory is None:
            return X, None
        scaler = factory()
        try:
            X_scaled = scaler.fit_transform(X)
        except Exception:
            # Fallback: standard scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        self._scaler = scaler
        return X_scaled, scaler

    def transform(self, X: np.ndarray, scaler: Any) -> np.ndarray:
        if scaler is None:
            return X
        return scaler.transform(X)

    @staticmethod
    def describe_scaler(scaler_type: ScalerType) -> str:
        descriptions = {
            ScalerType.STANDARD:      "Zero mean, unit variance (z-score normalization)",
            ScalerType.MINMAX:        "Scales to [0,1] range",
            ScalerType.ROBUST:        "Uses median and IQR — robust to outliers",
            ScalerType.QUANTILE_NORM: "Maps to normal distribution via quantiles",
            ScalerType.QUANTILE_UNIF: "Maps to uniform distribution via quantiles",
            ScalerType.POWER_YEO:     "Yeo-Johnson power transform — stabilises variance",
            ScalerType.POWER_BOX:     "Box-Cox power transform — requires positive values",
            ScalerType.MAXABS:        "Scales by max absolute value — preserves sign",
            ScalerType.NONE:          "No scaling applied",
        }
        return descriptions.get(scaler_type, "")


# ──────────────────────────────────────────────────────────────────
# CATEGORICAL ENCODER
# ──────────────────────────────────────────────────────────────────

class CategoricalEncoder:
    """Handles categorical columns for clustering preprocessing."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._encoders: Dict[str, LabelEncoder] = {}
        self._log: List[str] = []

    def fit_transform(self, df: pd.DataFrame,
                      cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self._log = []
        df = df.copy()
        encoders = {}

        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > self.config.max_categorical_cardinality:
                self._log.append(
                    f"Dropping '{col}': cardinality {n_unique} "
                    f"> max {self.config.max_categorical_cardinality}"
                )
                df.drop(columns=[col], inplace=True)
                continue

            if n_unique == 2:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                self._log.append(f"Binary-encoded '{col}'")
            else:
                # One-hot for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df.drop(columns=[col], inplace=True)
                df = pd.concat([df, dummies], axis=1)
                self._log.append(f"One-hot-encoded '{col}' → {len(dummies.columns)} columns")

        return df, encoders

    @property
    def log(self) -> List[str]:
        return self._log


# ──────────────────────────────────────────────────────────────────
# FEATURE SELECTOR
# ──────────────────────────────────────────────────────────────────

class FeatureSelector:
    """Dimensionality reduction and feature selection strategies."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._selector = None
        self._pca = None
        self._log: List[str] = []

    def fit_transform(self, X: np.ndarray,
                      feature_names: List[str]) -> Tuple[np.ndarray, List[str], Any]:
        self._log = []
        method = self.config.feature_selection

        if method == FeatureSelectionMethod.NONE:
            return X, feature_names, None

        if method == FeatureSelectionMethod.VARIANCE:
            return self._variance_selection(X, feature_names)
        elif method == FeatureSelectionMethod.CORRELATION:
            return self._correlation_selection(X, feature_names)
        elif method == FeatureSelectionMethod.PCA_REDUCE:
            return self._pca_reduction(X, feature_names)
        else:
            return X, feature_names, None

    def _variance_selection(self, X, feature_names):
        sel = VarianceThreshold(threshold=self.config.variance_threshold)
        X_new = sel.fit_transform(X)
        mask = sel.get_support()
        new_names = [n for n, m in zip(feature_names, mask) if m]
        n_removed = len(feature_names) - len(new_names)
        self._log.append(
            f"Variance filter: removed {n_removed} near-zero-variance features"
        )
        self._selector = sel
        return X_new, new_names, sel

    def _correlation_selection(self, X, feature_names):
        if X.shape[1] <= 2:
            return X, feature_names, None
        corr = np.corrcoef(X.T)
        to_drop = set()
        for i in range(len(corr)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(corr)):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > self.config.correlation_threshold:
                    to_drop.add(j)
        keep = [i for i in range(X.shape[1]) if i not in to_drop]
        X_new = X[:, keep]
        new_names = [feature_names[i] for i in keep]
        self._log.append(
            f"Correlation filter: kept {len(keep)}/{len(feature_names)} features "
            f"(threshold={self.config.correlation_threshold})"
        )
        return X_new, new_names, None

    def _pca_reduction(self, X, feature_names):
        n_components = self.config.pca_n_components
        if n_components is None:
            pca = PCA(n_components=self.config.pca_variance_explained,
                      svd_solver="full",
                      random_state=self.config.random_state)
        else:
            n_components = min(n_components, X.shape[1], X.shape[0] - 1)
            pca = PCA(n_components=n_components,
                      random_state=self.config.random_state)
        X_new = pca.fit_transform(X)
        new_names = [f"PC{i+1}" for i in range(X_new.shape[1])]
        var_exp = float(pca.explained_variance_ratio_.sum()) * 100
        self._log.append(
            f"PCA reduction: {X.shape[1]} → {X_new.shape[1]} components "
            f"({var_exp:.1f}% variance explained)"
        )
        self._pca = pca
        return X_new, new_names, pca

    @property
    def log(self) -> List[str]:
        return self._log


# ──────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """Master pipeline that orchestrates all preprocessing steps."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._loader = DataLoader()
        self._profiler = DataProfiler()
        self._imputer_handler = MissingValueHandler(self.config)
        self._outlier_detector = OutlierDetector(self.config)
        self._scaler = FeatureScaler(self.config.scaler_type)
        self._encoder = CategoricalEncoder(self.config)
        self._selector = FeatureSelector(self.config)
        self._log: List[str] = []
        self._warnings: List[str] = []

    def run(self, df: pd.DataFrame) -> PreprocessingResult:
        """Execute the full preprocessing pipeline."""
        self._log = []
        self._warnings = []
        X_original = df.copy()

        # Step 1: Profile raw data
        self._log.append("=== Step 1: Profiling raw data ===")
        profile = self._profiler.profile(df)
        self._log.append(
            f"Shape: {profile.n_rows}×{profile.n_cols} | "
            f"Missing: {profile.total_missing_pct:.1f}% | "
            f"Memory: {profile.memory_mb:.2f} MB"
        )

        # Step 2: Drop columns
        self._log.append("=== Step 2: Column selection ===")
        df, dropped = self._drop_columns(df, profile)

        # Step 3: Identify numeric / categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if self.config.numeric_columns is not None:
            numeric_cols = [c for c in self.config.numeric_columns if c in df.columns]
        if self.config.categorical_columns is not None:
            cat_cols = [c for c in self.config.categorical_columns if c in df.columns]

        self._log.append(
            f"Numeric: {len(numeric_cols)} cols | Categorical: {len(cat_cols)} cols"
        )

        # Step 4: Encode categoricals
        self._log.append("=== Step 4: Encoding categorical features ===")
        if cat_cols and self.config.encode_categoricals:
            df, _ = self._encoder.fit_transform(df, cat_cols)
            self._log.extend(self._encoder.log)

        # Refresh numeric cols after encoding
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric features remaining after preprocessing.")

        # Step 5: Missing value imputation
        self._log.append("=== Step 5: Imputing missing values ===")
        df, imputer = self._imputer_handler.fit_transform(df, numeric_cols)
        self._log.extend(self._imputer_handler.log)

        # Step 6: Outlier detection
        self._log.append("=== Step 6: Outlier detection ===")
        X_num = df[numeric_cols].values
        outlier_mask = self._outlier_detector.detect(X_num)
        n_outliers = int(outlier_mask.sum())
        self._log.extend(self._outlier_detector.log)

        if self.config.outlier_action != OutlierAction.NONE:
            df = self._outlier_detector.apply_action(df, outlier_mask, numeric_cols)
            # Refresh numeric cols in case __outlier__ was added
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "__outlier__"]

        # Step 7: Feature scaling
        self._log.append("=== Step 7: Scaling features ===")
        X_arr = df[numeric_cols].values
        X_scaled, scaler = self._scaler.fit_transform(X_arr)
        self._log.append(
            f"Applied {self.config.scaler_type.value} scaling to {len(numeric_cols)} features"
        )

        # Step 8: Feature selection
        self._log.append("=== Step 8: Feature selection ===")
        feature_names = list(numeric_cols)
        X_selected, feature_names, feature_selector = self._selector.fit_transform(
            X_scaled, feature_names
        )
        self._log.extend(self._selector.log)

        # Step 9: Final assembly
        X_processed = pd.DataFrame(X_selected, columns=feature_names)
        self._log.append(
            f"=== Final shape: {X_processed.shape[0]} × {X_processed.shape[1]} ==="
        )

        return PreprocessingResult(
            X_original=X_original,
            X_processed=X_processed,
            feature_names=feature_names,
            dropped_columns=dropped,
            outlier_mask=outlier_mask,
            n_outliers=n_outliers,
            scaler=scaler,
            imputer=imputer,
            feature_selector=feature_selector,
            config=self.config,
            profile=profile,
            log=self._log,
            warnings=self._warnings,
        )

    def _drop_columns(self, df: pd.DataFrame,
                      profile: DataProfile) -> Tuple[pd.DataFrame, List[str]]:
        dropped = list(self.config.drop_columns)

        # Auto-drop near-100% missing columns
        for col, cp in profile.column_profiles.items():
            if cp.missing_pct >= 99.0 and col not in dropped:
                dropped.append(col)
                self._log.append(f"Auto-dropping '{col}': {cp.missing_pct:.0f}% missing")

        # Auto-drop single-value columns
        for col in df.columns:
            if df[col].nunique() <= 1 and col not in dropped:
                dropped.append(col)
                self._log.append(f"Auto-dropping '{col}': zero variance")

        df = df.drop(columns=[c for c in dropped if c in df.columns])
        self._log.append(f"Dropped {len(dropped)} columns")
        return df, dropped

    @property
    def log(self) -> List[str]:
        return self._log


# ──────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def get_scaler_options() -> List[Dict[str, str]]:
    return [
        {"value": s.value, "label": s.value.replace("_", " ").title(),
         "description": FeatureScaler.describe_scaler(s)}
        for s in ScalerType
    ]


def get_imputer_options() -> List[Dict[str, str]]:
    labels = {
        ImputeStrategy.MEAN:      "Mean imputation",
        ImputeStrategy.MEDIAN:    "Median imputation (recommended)",
        ImputeStrategy.MODE:      "Mode (most frequent) imputation",
        ImputeStrategy.CONSTANT:  "Fill with constant value",
        ImputeStrategy.KNN:       "K-Nearest Neighbors imputation",
        ImputeStrategy.ITERATIVE: "Iterative (MICE) imputation",
        ImputeStrategy.DROP:      "Drop rows with missing values",
    }
    return [{"value": s.value, "label": labels[s]} for s in ImputeStrategy]


def get_outlier_options() -> List[Dict[str, str]]:
    labels = {
        OutlierMethod.ZSCORE:    "Z-score (threshold-based)",
        OutlierMethod.IQR:       "IQR (1.5× rule)",
        OutlierMethod.ISOLATION: "Isolation Forest",
        OutlierMethod.LOF:       "Local Outlier Factor",
        OutlierMethod.ELLIPTIC:  "Elliptic Envelope (Mahalanobis)",
        OutlierMethod.NONE:      "No outlier handling",
    }
    return [{"value": m.value, "label": labels[m]} for m in OutlierMethod]


def infer_best_config(profile: DataProfile) -> PreprocessingConfig:
    """Heuristic config recommendation based on data profile."""
    config = PreprocessingConfig()

    # Scaler recommendation
    skew_vals = [
        cp.skewness for cp in profile.column_profiles.values()
        if cp.skewness is not None
    ]
    if skew_vals:
        mean_abs_skew = np.mean(np.abs(skew_vals))
        if mean_abs_skew > 2.0:
            config.scaler_type = ScalerType.ROBUST
        elif mean_abs_skew > 0.8:
            config.scaler_type = ScalerType.QUANTILE_NORM
        else:
            config.scaler_type = ScalerType.STANDARD

    # Imputer recommendation
    if profile.total_missing_pct > 5.0:
        config.impute_strategy = ImputeStrategy.KNN
    elif profile.total_missing_pct > 0:
        config.impute_strategy = ImputeStrategy.MEDIAN

    # Outlier recommendation
    high_outlier_cols = sum(
        1 for cp in profile.column_profiles.values()
        if cp.outlier_pct > 5.0
    )
    if high_outlier_cols >= 2:
        config.outlier_method = OutlierMethod.ISOLATION
        config.outlier_action = OutlierAction.FLAG

    # Feature selection
    if profile.n_numeric > 50:
        config.feature_selection = FeatureSelectionMethod.PCA_REDUCE
        config.pca_variance_explained = 0.95
    elif len(profile.high_corr_pairs) > 5:
        config.feature_selection = FeatureSelectionMethod.CORRELATION
        config.correlation_threshold = 0.90

    return config


def summarize_preprocessing_result(result: PreprocessingResult) -> Dict[str, Any]:
    return {
        "original_shape": list(result.X_original.shape),
        "processed_shape": list(result.X_processed.shape),
        "dropped_columns": result.dropped_columns,
        "n_outliers": result.n_outliers,
        "outlier_pct": round(result.n_outliers / max(len(result.X_processed), 1) * 100, 2),
        "features": result.feature_names,
        "scaler": str(result.config.scaler_type.value),
        "imputer": str(result.config.impute_strategy.value),
        "outlier_method": str(result.config.outlier_method.value),
        "feature_selection": str(result.config.feature_selection.value),
        "warnings": result.warnings,
    }


# ══════════════════════════════════════════════════════════════════
# FINAL POLISH — ADVANCED PREPROCESSING ADDITIONS
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# FEATURE INTERACTION GENERATOR
# ──────────────────────────────────────────────────────────────────

class FeatureInteractionGenerator:
    """
    Generates polynomial / interaction features for clustering.
    Pairs of features multiplied or differenced can reveal
    structure invisible in original space.
    """
    def __init__(self, degree: int = 2, interaction_only: bool = True,
                 max_new_features: int = 50):
        self.degree = degree
        self.interaction_only = interaction_only
        self.max_new_features = max_new_features

    def fit_transform(self, X: np.ndarray,
                      feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        try:
            from sklearn.preprocessing import PolynomialFeatures
            n_orig = X.shape[1]
            max_feats = min(n_orig, 10)  # Cap to avoid explosion
            X_sub = X[:, :max_feats]
            names_sub = feature_names[:max_feats]
            poly = PolynomialFeatures(
                degree=self.degree,
                interaction_only=self.interaction_only,
                include_bias=False,
            )
            X_new = poly.fit_transform(X_sub)
            new_names = [
                n.replace(" ", "*") for n in
                poly.get_feature_names_out(names_sub)
            ]
            # Keep only new columns, cap
            X_interactions = X_new[:, n_orig:][:, :self.max_new_features]
            new_feature_names = new_names[n_orig:][:self.max_new_features]
            X_combined = np.hstack([X, X_interactions])
            all_names = feature_names + new_feature_names
            return X_combined, all_names
        except Exception as e:
            logger.warning(f"Interaction generation failed: {e}")
            return X, feature_names


# ──────────────────────────────────────────────────────────────────
# WHITENING TRANSFORM
# ──────────────────────────────────────────────────────────────────

class WhiteningTransform:
    """
    ZCA / PCA whitening: decorrelates features and normalises variance.
    Critical for distance-sensitive algorithms (K-Means, GMM, Spectral).
    ZCA whitening preserves the original feature space structure better than PCA.
    """
    def __init__(self, method: str = "zca", epsilon: float = 1e-5,
                 random_state: int = 42):
        assert method in ("zca", "pca"), "method must be 'zca' or 'pca'"
        self.method = method
        self.epsilon = epsilon
        self.random_state = random_state
        self._W = None
        self._mean = None

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, "WhiteningTransform"]:
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        cov = np.cov(Xc.T) + np.eye(X.shape[1]) * self.epsilon
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, self.epsilon)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
            if self.method == "zca":
                self._W = eigvecs @ D_inv_sqrt @ eigvecs.T
            else:  # pca
                self._W = D_inv_sqrt @ eigvecs.T
            X_white = (Xc @ self._W.T)
        except np.linalg.LinAlgError:
            logger.warning("Whitening eigdecomp failed; returning StandardScaler result")
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            X_white = ss.fit_transform(X)
        return X_white, self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._W is None or self._mean is None:
            raise RuntimeError("WhiteningTransform not fitted")
        return (X - self._mean) @ self._W.T


# ──────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION BENCHMARKER
# ──────────────────────────────────────────────────────────────────

class DimReducBenchmarker:
    """
    Compares multiple dimensionality reduction methods on the same data.
    Metric: trustworthiness (neighbourhood preservation) and
    reconstruction error (where applicable).
    Helps users choose the best embedding for their data.
    """
    METHODS = ["PCA", "ICA", "TruncatedSVD", "UMAP", "t-SNE"]

    def __init__(self, n_components: int = 10, random_state: int = 42,
                 max_samples: int = 5000):
        self.n_components = n_components
        self.random_state = random_state
        self.max_samples = max_samples

    def benchmark(self, X: np.ndarray,
                  methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        from sklearn.manifold import trustworthiness
        methods = methods or ["PCA", "ICA", "TruncatedSVD"]
        if len(X) > self.max_samples:
            idx = np.random.default_rng(self.random_state).choice(
                len(X), self.max_samples, replace=False)
            X_bench = X[idx]
        else:
            X_bench = X

        results = []
        nc = min(self.n_components, X_bench.shape[1] - 1, X_bench.shape[0] - 1)
        nc = max(nc, 2)

        for method in methods:
            t0 = time.perf_counter()
            try:
                if method == "PCA":
                    from sklearn.decomposition import PCA
                    m = PCA(n_components=nc, random_state=self.random_state)
                    Z = m.fit_transform(X_bench)
                    rec_err = float(np.mean((X_bench - m.inverse_transform(Z)) ** 2))
                    var_exp = float(m.explained_variance_ratio_.sum())
                elif method == "ICA":
                    from sklearn.decomposition import FastICA
                    m = FastICA(n_components=nc, random_state=self.random_state,
                                max_iter=300)
                    Z = m.fit_transform(X_bench)
                    rec_err = float(np.mean((X_bench - m.inverse_transform(Z)) ** 2))
                    var_exp = None
                elif method == "TruncatedSVD":
                    from sklearn.decomposition import TruncatedSVD
                    m = TruncatedSVD(n_components=nc, random_state=self.random_state)
                    Z = m.fit_transform(X_bench)
                    rec_err = float(np.mean((X_bench - m.inverse_transform(Z)) ** 2))
                    var_exp = float(m.explained_variance_ratio_.sum())
                elif method == "UMAP":
                    import umap
                    m = umap.UMAP(n_components=min(nc, 10),
                                  random_state=self.random_state)
                    Z = m.fit_transform(X_bench)
                    rec_err = None; var_exp = None
                elif method == "t-SNE":
                    from sklearn.manifold import TSNE
                    m = TSNE(n_components=min(nc, 3),
                             random_state=self.random_state,
                             n_iter=500, perplexity=min(30, len(X_bench)//4))
                    Z = m.fit_transform(X_bench)
                    rec_err = None; var_exp = None
                else:
                    continue

                # Trustworthiness (n_neighbors=12)
                try:
                    trust = float(trustworthiness(X_bench, Z, n_neighbors=12))
                except Exception:
                    trust = None

                runtime = time.perf_counter() - t0
                results.append({
                    "method": method,
                    "n_components": Z.shape[1],
                    "trustworthiness": round(trust, 4) if trust else None,
                    "reconstruction_error": round(rec_err, 4) if rec_err is not None else None,
                    "variance_explained": round(var_exp, 4) if var_exp is not None else None,
                    "runtime_seconds": round(runtime, 3),
                    "status": "ok",
                })
            except Exception as e:
                results.append({
                    "method": method, "status": "failed", "error": str(e),
                    "runtime_seconds": round(time.perf_counter() - t0, 3),
                })

        return sorted(results,
                      key=lambda r: r.get("trustworthiness") or 0,
                      reverse=True)


# ──────────────────────────────────────────────────────────────────
# SMART DATA TYPE DETECTOR
# ──────────────────────────────────────────────────────────────────

class SmartDataTypeDetector:
    """
    Detects special data characteristics that affect algorithm choice:
    - Time-series columns (monotonic, periodic patterns)
    - Text-derived columns (high cardinality, string lengths)
    - ID-like columns (near-unique, sequential integers)
    - Constant / near-constant columns
    - Highly skewed columns needing log transform
    - Bimodal columns (suitable for GMM)
    """
    def detect(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        results = {}
        for col in df.columns:
            series = df[col]
            info: Dict[str, Any] = {"col": col, "flags": []}
            n_unique = series.nunique()
            n = len(series)

            # ID-like
            if series.dtype in [np.int64, np.int32, np.float64]:
                if n_unique / max(n, 1) > 0.95:
                    info["flags"].append("id_like")

            # Constant
            if n_unique <= 1:
                info["flags"].append("constant")

            # Numeric analysis
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 10:
                    skew = float(clean.skew())
                    if abs(skew) > 2.5:
                        info["flags"].append("highly_skewed")
                        info["skewness"] = round(skew, 3)
                        if clean.min() > 0:
                            info["flags"].append("log_transform_candidate")

                    # Bimodality test (Hartigan's dip approximation via kurtosis)
                    kurt = float(clean.kurtosis())
                    if kurt < -0.5:
                        info["flags"].append("possibly_bimodal")

                    # Monotonic (time-series proxy)
                    diffs = np.diff(clean.values)
                    if (diffs > 0).mean() > 0.98 or (diffs < 0).mean() > 0.98:
                        info["flags"].append("monotonic_time_series")

                    # Periodic (autocorrelation test)
                    if len(clean) >= 20:
                        try:
                            ac = float(pd.Series(clean.values).autocorr(lag=1))
                            if abs(ac) > 0.85:
                                info["flags"].append("high_autocorrelation")
                                info["autocorr_lag1"] = round(ac, 3)
                        except Exception:
                            pass

            results[col] = info

        # Summary
        all_flags = [f for info in results.values() for f in info.get("flags", [])]
        from collections import Counter
        return {
            "columns": results,
            "flag_summary": dict(Counter(all_flags)),
            "n_id_like": sum(1 for i in results.values() if "id_like" in i.get("flags", [])),
            "n_skewed": sum(1 for i in results.values() if "highly_skewed" in i.get("flags", [])),
            "n_bimodal": sum(1 for i in results.values() if "possibly_bimodal" in i.get("flags", [])),
            "n_timeseries": sum(1 for i in results.values() if "monotonic_time_series" in i.get("flags", [])),
            "recommendations": _preprocessing_recommendations(results),
        }


def _preprocessing_recommendations(col_results: Dict) -> List[str]:
    recs = []
    n_skewed = sum(1 for i in col_results.values() if "highly_skewed" in i.get("flags",[]))
    n_log    = sum(1 for i in col_results.values() if "log_transform_candidate" in i.get("flags",[]))
    n_mono   = sum(1 for i in col_results.values() if "monotonic_time_series" in i.get("flags",[]))
    n_bim    = sum(1 for i in col_results.values() if "possibly_bimodal" in i.get("flags",[]))
    n_id     = sum(1 for i in col_results.values() if "id_like" in i.get("flags",[]))
    if n_skewed > 0:
        recs.append(f"⚠️ {n_skewed} highly skewed feature(s) — use Robust or Quantile scaler.")
    if n_log > 0:
        recs.append(f"✅ {n_log} positive skewed feature(s) are log-transform candidates — try PowerTransformer (Yeo-Johnson).")
    if n_mono > 0:
        recs.append(f"⏱️ {n_mono} monotonic column(s) detected — likely time/ID columns; consider dropping or differencing.")
    if n_bim > 0:
        recs.append(f"🔀 {n_bim} possibly bimodal feature(s) — GMM or DPGMM may work better than K-Means here.")
    if n_id > 0:
        recs.append(f"🆔 {n_id} near-unique (ID-like) column(s) detected — strongly recommend dropping these.")
    if not recs:
        recs.append("✅ No major data quality flags. Standard preprocessing should work well.")
    return recs


# ──────────────────────────────────────────────────────────────────
# MULTI-COLLINEARITY REMOVER (VIF-based)
# ──────────────────────────────────────────────────────────────────

class VIFCollinearityRemover:
    """
    Variance Inflation Factor (VIF) based collinearity removal.
    More principled than simple correlation filtering because it
    accounts for multicollinearity (not just pairwise correlation).
    Iteratively removes the feature with highest VIF until all VIF < threshold.
    """
    def __init__(self, vif_threshold: float = 10.0, max_features: int = 200):
        self.vif_threshold = vif_threshold
        self.max_features = max_features
        self._dropped: List[str] = []

    def fit_transform(self, X: np.ndarray,
                      feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        if X.shape[1] > self.max_features:
            return X, feature_names

        remaining = list(range(X.shape[1]))
        names = list(feature_names)
        self._dropped = []
        max_iter = X.shape[1]

        for _ in range(max_iter):
            if len(remaining) <= 2:
                break
            Xs = X[:, remaining]
            vifs = self._compute_vifs(Xs)
            max_vif = float(vifs.max())
            if max_vif < self.vif_threshold:
                break
            worst = int(vifs.argmax())
            dropped_name = names[worst]
            self._dropped.append(dropped_name)
            remaining.pop(worst)
            names.pop(worst)

        return X[:, remaining], names

    @staticmethod
    def _compute_vifs(X: np.ndarray) -> np.ndarray:
        try:
            from sklearn.linear_model import LinearRegression
            n, p = X.shape
            vifs = np.zeros(p)
            for j in range(p):
                y = X[:, j]
                Xj = np.delete(X, j, axis=1)
                lr = LinearRegression(fit_intercept=True)
                lr.fit(Xj, y)
                ss_res = float(np.sum((y - lr.predict(Xj)) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = max(0.0, 1 - ss_res / (ss_tot + 1e-10))
                vifs[j] = 1.0 / (1 - r2 + 1e-10)
            return vifs
        except Exception:
            return np.ones(X.shape[1])

    @property
    def dropped_features(self) -> List[str]:
        return self._dropped


# ──────────────────────────────────────────────────────────────────
# ADAPTIVE SAMPLE WEIGHTER
# ──────────────────────────────────────────────────────────────────

class AdaptiveSampleWeighter:
    """
    Assigns sample weights to counteract density imbalance.
    Dense regions get lower weights; sparse regions get higher weights.
    Helps density-sensitive algorithms (GMM, K-Means) avoid being
    dominated by a high-density region.
    """
    def __init__(self, method: str = "inverse_density", n_neighbors: int = 10):
        assert method in ("inverse_density", "uniform", "log_inverse")
        self.method = method
        self.n_neighbors = n_neighbors

    def compute_weights(self, X: np.ndarray) -> np.ndarray:
        if self.method == "uniform":
            return np.ones(len(X))
        try:
            from sklearn.neighbors import NearestNeighbors
            k = min(self.n_neighbors, len(X) - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
            dists, _ = nbrs.kneighbors(X)
            # kth-NN distance as density proxy
            knn_dist = dists[:, -1]
            knn_dist = np.where(knn_dist < 1e-10, 1e-10, knn_dist)
            if self.method == "inverse_density":
                weights = knn_dist / knn_dist.mean()
            else:  # log_inverse
                weights = np.log1p(knn_dist) / np.log1p(knn_dist.mean())
            weights = weights / weights.sum() * len(X)
            return weights.astype(np.float32)
        except Exception:
            return np.ones(len(X), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────
# PREPROCESSING COMPARISON UTILITY
# ──────────────────────────────────────────────────────────────────

def compare_scalers(X: np.ndarray,
                    feature_idx: int = 0) -> Dict[str, np.ndarray]:
    """
    Returns the distribution of a single feature under each scaler.
    Useful for choosing the right scaler for skewed data.
    """
    results = {}
    scalers_to_try = {
        "Raw": None,
        "Standard": ScalerType.STANDARD,
        "MinMax": ScalerType.MINMAX,
        "Robust": ScalerType.ROBUST,
        "QuantileNorm": ScalerType.QUANTILE_NORM,
        "PowerYeo": ScalerType.POWER_YEO,
    }
    col = X[:, feature_idx:feature_idx+1]
    for name, st in scalers_to_try.items():
        if st is None:
            results[name] = col.ravel()
        else:
            try:
                fs = FeatureScaler(st)
                scaled, _ = fs.fit_transform(col)
                results[name] = scaled.ravel()
            except Exception:
                pass
    return results


def auto_detect_and_recommend(df: pd.DataFrame,
                               profile: Optional["DataProfile"] = None
                               ) -> Dict[str, Any]:
    """
    One-shot function: detect data types, run profiler, return full
    recommendations dict for the UI.
    """
    detector = SmartDataTypeDetector()
    detection = detector.detect(df)

    recommendations = {
        "detection": detection,
        "flag_summary": detection.get("flag_summary", {}),
        "preprocessing_recommendations": detection.get("recommendations", []),
    }

    if profile is not None:
        cfg = infer_best_config(profile)
        recommendations["suggested_config"] = {
            "scaler": cfg.scaler_type.value,
            "imputer": cfg.impute_strategy.value,
            "outlier_method": cfg.outlier_method.value,
            "feature_selection": cfg.feature_selection.value,
        }

    return recommendations
