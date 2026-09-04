import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)


FEATURE_COLUMNS = [
    "amount",
    "transaction_hour",
    "customer_frequency",
    "merchant_frequency",
    "failed_attempts",
    "device_change",
    "location_change",
    "amount_deviation",
    "velocity_1h",
    "velocity_24h",
    "previous_fraud_count",
    "device_customer_degree",
    "merchant_customer_degree",
    "device_reuse_ratio",
    "recent_device_velocity",
    "customer_merchant_count",
]


BASE_COLUMNS = [
    "timestamp",
    "transaction_id",
    "customer_id",
    "merchant_id",
    "amount",
    "device_id",
    "location",
    "failed_attempts",
    "device_changed",
    "location_changed",
    "is_fraud",
]


class RiskEngine:

    def __init__(self, data_path=None, artifact_dir=None):

        base = Path(__file__).resolve().parent.parent

        self.data_path = Path(
            data_path or base / "data" / "transactions.csv"
        )

        self.artifact_dir = Path(
            artifact_dir or base / "artifacts"
        )

        self.artifact_dir.mkdir(exist_ok=True)

        self.model_path = (
            self.artifact_dir / "riskshield_models.joblib"
        )

        self.graph_path = (
            self.artifact_dir / "graph_stats.json"
        )

        # Load existing trained artifacts if available.
        if self.model_path.exists() and self.graph_path.exists():

            try:

                bundle = joblib.load(self.model_path)

                self.classifier = bundle["classifier"]
                self.anomaly = bundle["anomaly"]
                self.amin = bundle["amin"]
                self.amax = bundle["amax"]
                self.background = bundle["background"]

                with open(self.graph_path, "r") as f:
                    self.stats = json.load(f)

                return

            except Exception:
                pass

        # Train if artifacts are unavailable.
        self.train()

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------

    def load_data(self):

        df = pd.read_csv(
            self.data_path,
            parse_dates=["timestamp"]
        )

        df = (
            df.sort_values("timestamp")
            .reset_index(drop=True)
        )

        df["device_changed"] = (
            df["device_changed"]
            .astype(int)
        )

        df["location_changed"] = (
            df["location_changed"]
            .astype(int)
        )

        df["device_change"] = df["device_changed"]

        df["location_change"] = df["location_changed"]

        return df

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------

    def build_features(self, df):

        df = df.copy()

        df["device_change"] = (
            df["device_changed"]
            .astype(int)
        )

        df["location_change"] = (
            df["location_changed"]
            .astype(int)
        )

        df["transaction_hour"] = (
            df["timestamp"].dt.hour
        )

        # --------------------------------------------------------------
        # Causal customer amount statistics
        # --------------------------------------------------------------

        df["cust_running_mean"] = (
            df.groupby("customer_id")["amount"]
            .transform(
                lambda s:
                s.shift(1)
                .rolling(20, min_periods=1)
                .mean()
            )
        )

        df["cust_running_std"] = (
            df.groupby("customer_id")["amount"]
            .transform(
                lambda s:
                s.shift(1)
                .rolling(20, min_periods=2)
                .std()
            )
            .fillna(1.0)
        )

        df["amount_deviation"] = (
            (
                df["amount"]
                - df["cust_running_mean"]
                .fillna(df["amount"])
            )
            /
            (
                df["cust_running_std"]
                .replace(0, np.nan)
                .fillna(1.0)
            )
        )

        # --------------------------------------------------------------
        # Frequency features
        # --------------------------------------------------------------

        df["customer_frequency"] = (
            df.groupby("customer_id")
            .cumcount()
        )

        df["merchant_frequency"] = (
            df.groupby("merchant_id")
            .cumcount()
        )

        # --------------------------------------------------------------
        # Transaction velocity
        # --------------------------------------------------------------

        df["velocity_1h"] = (
            self._velocity(df, "1h")
        )

        df["velocity_24h"] = (
            self._velocity(df, "24h")
        )

        # --------------------------------------------------------------
        # Previous fraud count
        # --------------------------------------------------------------

        df["previous_fraud_count"] = (
            df.groupby("customer_id")["is_fraud"]
            .transform(
                lambda s:
                s.shift(1)
                .fillna(0)
                .cumsum()
            )
        )

        # --------------------------------------------------------------
        # Temporal graph features
        # --------------------------------------------------------------

        graph_features = self._graph_features(df)

        for column in graph_features.columns:
            df[column] = graph_features[column].values

        return df

    # ------------------------------------------------------------------
    # VELOCITY
    # ------------------------------------------------------------------

    def _velocity(self, df, window):

        seconds = (
            pd.Timedelta(window)
            .total_seconds()
        )

        out = np.ones(
            len(df),
            dtype=float
        )

        last_by_customer = {}

        times = (
            df["timestamp"]
            .astype("int64")
            .values
            / 1e9
        )

        customers = (
            df["customer_id"]
            .values
        )

        for i, (t, customer) in enumerate(
            zip(times, customers)
        ):

            arr = last_by_customer.get(
                customer,
                []
            )

            cutoff = t - seconds

            while arr and arr[0] <= cutoff:
                arr.pop(0)

            out[i] = len(arr) + 1

            arr.append(t)

            last_by_customer[customer] = arr

        return out

    # ------------------------------------------------------------------
    # GRAPH FEATURES
    # ------------------------------------------------------------------

    def _graph_features(self, df):

        device_customers = {}
        merchant_customers = {}
        cust_merchants = {}
        recent_device_times = {}

        rows = []

        for _, r in df.iterrows():

            dev = r.device_id
            cust = r.customer_id
            merch = r.merchant_id

            dc = device_customers.get(
                dev,
                set()
            )

            mc = merchant_customers.get(
                merch,
                set()
            )

            cm = cust_merchants.get(
                cust,
                set()
            )

            prior_dev_count = len(dc)

            prior_merchant_count = len(mc)

            prior_recent = len([
                t
                for t in recent_device_times.get(
                    dev,
                    []
                )
                if r.timestamp.value / 1e9 - t < 3600
            ])

            rows.append({
                "device_customer_degree":
                    prior_dev_count,

                "merchant_customer_degree":
                    prior_merchant_count,

                "device_reuse_ratio":
                    prior_dev_count /
                    (prior_dev_count + 1),

                "recent_device_velocity":
                    prior_recent,

                "customer_merchant_count":
                    len(cm),
            })

            dc.add(cust)
            device_customers[dev] = dc

            mc.add(cust)
            merchant_customers[merch] = mc

            cm.add(merch)
            cust_merchants[cust] = cm

            recent_device_times.setdefault(
                dev,
                []
            ).append(
                r.timestamp.value / 1e9
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train(self):

        df = self.build_features(
            self.load_data()
        )

        # Time split to avoid leakage.
        split = int(len(df) * 0.80)

        train = df.iloc[:split]

        valid = df.iloc[split:]

        X = (
            train[FEATURE_COLUMNS]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        y = (
            train["is_fraud"]
            .astype(int)
        )

        # --------------------------------------------------------------
        # Random Forest
        # --------------------------------------------------------------

        self.classifier = RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

        self.classifier.fit(
            X,
            y
        )

        # --------------------------------------------------------------
        # Isolation Forest
        # --------------------------------------------------------------

        self.anomaly = IsolationForest(
            n_estimators=180,
            contamination=min(
                0.03,
                max(
                    0.005,
                    float(y.mean())
                )
            ),
            random_state=42,
            n_jobs=-1,
        )

        self.anomaly.fit(X)

        raw = (
            -self.anomaly.score_samples(X)
        )

        self.amin = float(
            raw.min()
        )

        self.amax = float(
            raw.max()
        )

        # Background data for explanations.
        self.background = (
            train[FEATURE_COLUMNS]
            .sample(
                min(
                    2500,
                    len(train)
                ),
                random_state=42,
            )
            .reset_index(drop=True)
        )

        # --------------------------------------------------------------
        # Model statistics
        # --------------------------------------------------------------

        self.stats = {
            "customers":
                int(df.customer_id.nunique()),

            "merchants":
                int(df.merchant_id.nunique()),

            "devices":
                int(df.device_id.nunique()),

            "transactions":
                int(len(df)),

            "fraud_count":
                int(df.is_fraud.sum()),

            "fraud_rate":
                float(df.is_fraud.mean()),

            "feature_columns":
                FEATURE_COLUMNS,

            "trained_rows":
                int(len(train)),

            "validation_rows":
                int(len(valid)),

            "model":
                "RandomForest + IsolationForest + Temporal Risk Graph",
        }

        # --------------------------------------------------------------
        # Validation metrics
        # --------------------------------------------------------------

        Xv = (
            valid[FEATURE_COLUMNS]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        yv = (
            valid["is_fraud"]
            .astype(int)
        )

        pv = (
            self.classifier
            .predict_proba(Xv)[:, 1]
        )

        if yv.nunique() > 1:

            self.stats["validation_auc"] = float(
                roc_auc_score(
                    yv,
                    pv
                )
            )

            self.stats["validation_pr_auc"] = float(
                average_precision_score(
                    yv,
                    pv
                )
            )

        else:

            self.stats["validation_auc"] = None
            self.stats["validation_pr_auc"] = None

        pred = (
            pv >= 0.5
        ).astype(int)

        p, r, f, _ = (
            precision_recall_fscore_support(
                yv,
                pred,
                average="binary",
                zero_division=0,
            )
        )

        self.stats["validation_precision"] = float(p)
        self.stats["validation_recall"] = float(r)
        self.stats["validation_f1"] = float(f)

        # --------------------------------------------------------------
        # Save artifacts
        # --------------------------------------------------------------

        joblib.dump(
            {
                "classifier":
                    self.classifier,

                "anomaly":
                    self.anomaly,

                "amin":
                    self.amin,

                "amax":
                    self.amax,

                "background":
                    self.background,
            },
            self.model_path,
        )

        with open(
            self.graph_path,
            "w"
        ) as f:

            json.dump(
                self.stats,
                f,
                indent=2
            )

    # ------------------------------------------------------------------
    # ANOMALY SCORE
    # ------------------------------------------------------------------

    def _anomaly_score(self, X):

        raw = (
            -self.anomaly
            .score_samples(X)
        )

        return float(
            np.clip(
                (
                    raw[0]
                    - self.amin
                )
                /
                (
                    self.amax
                    - self.amin
                    + 1e-9
                ),
                0,
                1,
            )
        )

    # ------------------------------------------------------------------
    # LIVE GRAPH SCORE
    # ------------------------------------------------------------------

    def _graph_live_score(
        self,
        payload
    ):

        ring = float(
            payload.get(
                "ring_score",
                0
            )
            or 0
        )

        signals = {}

        if (
            payload.get("device_id")
            in self.stats.get(
                "high_risk_devices",
                []
            )
        ):

            ring = max(
                ring,
                0.8
            )

        if payload.get(
            "device_changed"
        ):

            ring = max(
                ring,
                0.35
            )

        if payload.get(
            "location_changed"
        ):

            ring = max(
                ring,
                0.25
            )

        if (
            payload.get(
                "failed_attempts",
                0
            )
            >= 3
        ):

            ring = max(
                ring,
                0.30
            )

        if (
            payload.get(
                "velocity_1h"
            )
            is not None
            and payload["velocity_1h"] >= 6
        ):

            ring = max(
                ring,
                0.55
            )

        signals[
            "shared_device_or_network"
        ] = round(
            ring,
            3
        )

        signals["device_changed"] = bool(
            payload.get(
                "device_changed",
                False
            )
        )

        signals["location_changed"] = bool(
            payload.get(
                "location_changed",
                False
            )
        )

        signals["failed_attempts"] = (
            payload.get(
                "failed_attempts",
                0
            )
        )

        return (
            float(
                np.clip(
                    ring,
                    0,
                    1
                )
            ),
            signals,
        )

    # ------------------------------------------------------------------
    # PAYLOAD FEATURES
    # ------------------------------------------------------------------

    def payload_features(self, p):

        ts = (
            pd.Timestamp(
                p.get("timestamp")
            )
            if p.get("timestamp")
            else pd.Timestamp.utcnow()
        )

        amount = float(
            p.get(
                "amount",
                1000
            )
        )

        avg = p.get(
            "customer_avg_amount"
        )

        if avg:

            amount_dev = (
                amount - float(avg)
            ) / max(
                abs(float(avg)) * 0.35,
                1
            )

        else:

            amount_dev = (
                amount - 1000
            ) / 1000

        row = {

            "amount":
                amount,

            "transaction_hour":
                int(ts.hour),

            "customer_frequency":
                int(
                    p.get(
                        "customer_frequency",
                        10
                    )
                ),

            "merchant_frequency":
                int(
                    p.get(
                        "merchant_frequency",
                        100
                    )
                ),

            "failed_attempts":
                int(
                    p.get(
                        "failed_attempts",
                        0
                    )
                ),

            "device_change":
                int(
                    bool(
                        p.get(
                            "device_changed",
                            False
                        )
                    )
                ),

            "location_change":
                int(
                    bool(
                        p.get(
                            "location_changed",
                            False
                        )
                    )
                ),

            "amount_deviation":
                amount_dev,

            "velocity_1h": float(p.get("velocity_1h") if p.get("velocity_1h") is not None else 1),

            "velocity_24h": float(p.get("velocity_24h") if p.get("velocity_24h") is not None else 3),

            "previous_fraud_count": float(p.get("previous_fraud_count") if p.get("previous_fraud_count") is not None else 0),

            "device_customer_degree": float(p.get("device_customer_degree") if p.get("device_customer_degree") is not None else 0),

            "merchant_customer_degree": float(p.get("merchant_customer_degree") if p.get("merchant_customer_degree") is not None else 0),

            "device_reuse_ratio": float(p.get("device_reuse_ratio") if p.get("device_reuse_ratio") is not None else 0),

            "recent_device_velocity": float(p.get("recent_device_velocity") if p.get("recent_device_velocity") is not None else 0),

            "customer_merchant_count": float(p.get("customer_merchant_count") if p.get("customer_merchant_count") is not None else 1),
        }

        return pd.DataFrame(
            [row]
        )

    # ------------------------------------------------------------------
    # EXPLAINABILITY
    # ------------------------------------------------------------------

    def explain(self, row):

        X = (
            row[FEATURE_COLUMNS]
            .values
        )

        base = float(
            self.classifier
            .predict_proba(
                self.background[
                    FEATURE_COLUMNS
                ]
            )[:, 1]
            .mean()
        )

        contrib = {}

        original = float(
            self.classifier
            .predict_proba(X)[0, 1]
        )

        for i, c in enumerate(
            FEATURE_COLUMNS
        ):

            pert = X.copy()

            pert[0, i] = float(
                self.background[c]
                .median()
            )

            contrib[c] = (
                original
                - float(
                    self.classifier
                    .predict_proba(
                        pert
                    )[0, 1]
                )
            )

        ranked = sorted(
            contrib.items(),
            key=lambda x:
                abs(x[1]),
            reverse=True,
        )

        reasons = []

        for c, v in ranked[:4]:

            val = float(
                row[c].iloc[0]
            )

            if abs(v) < 0.01:
                continue

            direction = (
                "increasing"
                if v > 0
                else "reducing"
            )

            reasons.append(
                f"{c.replace('_', ' ')} "
                f"is {val:.2f}, "
                f"{direction} model risk"
            )

        return (
            reasons
            or [
                "No single feature dominates "
                "the model decision."
            ]
        )

    # ------------------------------------------------------------------
    # COUNTERFACTUAL
    # ------------------------------------------------------------------

    def counterfactual(
        self,
        row,
        threshold=0.70
    ):

        current = row.copy()

        start = float(
            current["amount"].iloc[0]
        )

        if float(
            self.classifier
            .predict_proba(
                current[FEATURE_COLUMNS]
            )[0, 1]
        ) < threshold:

            return None

        for _ in range(15):

            current.loc[:, "amount"] *= 0.9

            current.loc[
                :,
                "amount_deviation"
            ] *= 0.9

            if float(
                self.classifier
                .predict_proba(
                    current[
                        FEATURE_COLUMNS
                    ]
                )[0, 1]
            ) < threshold:

                return (
                    "Illustrative counterfactual: "
                    f"reduce amount from "
                    f"{start:.0f} to "
                    f"{float(current.amount.iloc[0]):.0f}."
                )

        return (
            "The model remained above the "
            "high-risk threshold after 15 "
            "simulated amount reductions."
        )

    # ------------------------------------------------------------------
    # INDIVIDUAL TRANSACTION SCORE
    # ------------------------------------------------------------------

    def score(self, payload):

        row = self.payload_features(
            payload
        )

        X = row[
            FEATURE_COLUMNS
        ]

        fraud = float(
            self.classifier
            .predict_proba(X)[0, 1]
        )

        anomaly = (
            self._anomaly_score(X)
        )

        ring, signals = (
            self._graph_live_score(
                payload
            )
        )

        # Graph-aware ensemble.
        combined = (
            0.50 * fraud
            + 0.25 * anomaly
            + 0.25 * ring
        )

        combined = float(
            np.clip(
                combined,
                0,
                1
            )
        )

        score = int(
            round(
                combined * 100
            )
        )

        if combined >= 0.70:

            level = "HIGH"
            decision = "MANUAL_REVIEW"

        elif combined >= 0.40:

            level = "MEDIUM"
            decision = "ADDITIONAL_VERIFICATION"

        else:

            level = "LOW"
            decision = "ALLOW"

        result = {

            "transaction_id":
                payload.get(
                    "transaction_id",
                    "UNKNOWN"
                ),

            "risk_score":
                score,

            "risk_level":
                level,

            "fraud_probability":
                round(
                    fraud,
                    4
                ),

            "anomaly_score":
                round(
                    anomaly,
                    4
                ),

            "ring_score":
                round(
                    ring,
                    4
                ),

            "combined_score":
                round(
                    combined,
                    4
                ),

            "decision":
                decision,

            "reasons":
                self.explain(row),

            "counterfactual":
                (
                    self.counterfactual(
                        row,
                        0.70
                    )
                    if level == "HIGH"
                    else None
                ),

            "graph_signals":
                signals,
        }

        return result

    # ------------------------------------------------------------------
    # FAST REFERENCE EVALUATION
    # ------------------------------------------------------------------

    def evaluate_reference(
        self,
        test_path
    ):
        """
        Fast reference-set evaluation.

        IMPORTANT:
        Do not call self.score() here.

        self.score() performs expensive local
        explanation and counterfactual calculations.
        Those calculations are useful for a single
        transaction but are unnecessary for aggregate
        evaluation metrics.

        This method therefore performs batch model
        predictions instead.
        """

        test = pd.read_csv(
            test_path,
            parse_dates=["timestamp"]
        )

        if "is_fraud" not in test.columns:
            return {}

        # --------------------------------------------------------------
        # Build features once
        # --------------------------------------------------------------

        test_features = (
            self.build_features(test)
        )

        X = (
            test_features[
                FEATURE_COLUMNS
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        y = (
            test_features["is_fraud"]
            .astype(int)
            .values
        )

        # --------------------------------------------------------------
        # Batch Random Forest prediction
        # --------------------------------------------------------------

        fraud_probs = (
            self.classifier
            .predict_proba(X)[:, 1]
        )

        # --------------------------------------------------------------
        # Batch Isolation Forest prediction
        # --------------------------------------------------------------

        raw_anomaly = (
            -self.anomaly
            .score_samples(X)
        )

        anomaly_scores = np.clip(
            (
                raw_anomaly
                - self.amin
            )
            /
            (
                self.amax
                - self.amin
                + 1e-9
            ),
            0,
            1,
        )

        # --------------------------------------------------------------
        # Graph signal
        # --------------------------------------------------------------

        ring = np.zeros(
            len(test_features),
            dtype=float
        )

        if "ring_score" in test_features.columns:

            ring = (
                test_features["ring_score"]
                .fillna(0)
                .astype(float)
                .values
            )

        if "device_changed" in test_features.columns:

            ring = np.maximum(
                ring,
                (
                    test_features[
                        "device_changed"
                    ]
                    .astype(int)
                    .values
                    * 0.35
                )
            )

        if "location_changed" in test_features.columns:

            ring = np.maximum(
                ring,
                (
                    test_features[
                        "location_changed"
                    ]
                    .astype(int)
                    .values
                    * 0.25
                )
            )

        if "failed_attempts" in test_features.columns:

            ring = np.maximum(
                ring,
                np.where(
                    (
                        test_features[
                            "failed_attempts"
                        ]
                        .astype(float)
                        .values
                        >= 3
                    ),
                    0.30,
                    0.0,
                )
            )

        if "velocity_1h" in test_features.columns:

            ring = np.maximum(
                ring,
                np.where(
                    (
                        test_features[
                            "velocity_1h"
                        ]
                        .astype(float)
                        .values
                        >= 6
                    ),
                    0.55,
                    0.0,
                )
            )

        ring = np.clip(
            ring,
            0,
            1
        )

        # --------------------------------------------------------------
        # Same ensemble weighting as score()
        # --------------------------------------------------------------

        combined = (
            0.50 * fraud_probs
            + 0.25 * anomaly_scores
            + 0.25 * ring
        )

        combined = np.clip(
            combined,
            0,
            1
        )

        # --------------------------------------------------------------
        # Evaluation metrics
        # --------------------------------------------------------------

        unique_labels = np.unique(y)

        if len(unique_labels) > 1:

            ensemble_auc = float(
                roc_auc_score(
                    y,
                    combined
                )
            )

            ensemble_pr_auc = float(
                average_precision_score(
                    y,
                    combined
                )
            )

        else:

            ensemble_auc = None
            ensemble_pr_auc = None

        return {

            "rows":
                int(
                    len(test_features)
                ),

            "fraud_count":
                int(
                    y.sum()
                ),

            "fraud_rate":
                float(
                    y.mean()
                ),

            "ensemble_auc":
                ensemble_auc,

            "ensemble_pr_auc":
                ensemble_pr_auc,
        }

