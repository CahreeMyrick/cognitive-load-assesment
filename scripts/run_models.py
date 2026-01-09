# scripts/run_models.py

from __future__ import annotations

import argparse
import os
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.utils import load_config, ensure_dir, downsample_dataframe


# ============================================================
# Allowed experiment modes 
# ============================================================

ALLOWED_MODES = {
    "eeg": ("EEG",),
    "gsr": ("GSR",),
    "ppg": ("PPG",),
    "gsr_ppg": ("GSR", "PPG"),
}


# ============================================================
# Dataset Loader
# ============================================================

class Dataset:
    def __init__(self, cfg: dict, modalities: tuple[str, ...]):
        self.cfg = cfg
        self.modalities = modalities
        self.outdir = cfg.get("outputs_dir", "outputs")
        self.X, self.y = self._load()

    def _load(self):
        if len(self.modalities) == 1:
            return self._load_single(self.modalities[0])
        return self._load_fusion()

    def _load_single(self, modality: str):
        X = pd.read_csv(os.path.join(self.outdir, f"{modality}features.csv"))
        y = pd.read_csv(
            os.path.join(self.outdir, f"{modality}labels.csv")
        ).squeeze()
        print(f"[LOAD] {modality}: X={X.shape}, y={y.shape}")
        return X, y

    def _load_fusion(self):
        cache, lengths = {}, []

        for mod in self.modalities:
            X, y = self._load_single(mod)
            cache[mod] = (X, y)
            lengths.append(len(X))

        target_len = min(lengths)
        print(f"[FUSION] Target aligned length = {target_len}")

        X_parts, y_ref = [], None
        for mod in self.modalities:
            X, y = cache[mod]
            X = downsample_dataframe(X, target_len)
            X_parts.append(X)
            if y_ref is None:
                y_ref = y.iloc[:target_len]

        X = pd.concat(X_parts, axis=1)
        y = y_ref.reset_index(drop=True)
        return X, y


# ============================================================
# Experiment Runner
# ============================================================

class Experiment:
    def __init__(self, cfg: dict, mode: str, seed: int):
        self.cfg = cfg
        self.mode = mode
        self.modalities = ALLOWED_MODES[mode]
        self.seed = seed

        self.exp_id = f"{mode}_{uuid.uuid4().hex[:8]}"
        self.timestamp = datetime.now().isoformat(timespec="seconds")

        self.outdir = os.path.join(
            cfg["outputs_dir"],
            "experiments",
            self.exp_id,
        )
        ensure_dir(self.outdir)

        self.dataset = Dataset(cfg, self.modalities)
        self.results = []

    def make_pipeline(self, clf):
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                ("clf", clf),
            ]
        )

    def run(self, clf_name: str, clf, n_splits: int = 5):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.seed
        )

        accs = []

        print(f"\n[RUN] {clf_name} | mode={self.mode}")
        for fold, (tr, te) in enumerate(
            skf.split(self.dataset.X, self.dataset.y), 1
        ):
            Xtr, Xte = (
                self.dataset.X.iloc[tr],
                self.dataset.X.iloc[te],
            )
            ytr, yte = (
                self.dataset.y.iloc[tr],
                self.dataset.y.iloc[te],
            )

            pipe = self.make_pipeline(clf)
            pipe.fit(Xtr, ytr)
            ypred = pipe.predict(Xte)

            acc = accuracy_score(yte, ypred)
            accs.append(acc)

            print(f"  Fold {fold}: acc={acc:.4f}")

        result = {
            "experiment_id": self.exp_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "modalities": "+".join(self.modalities),
            "classifier": clf_name,
            "seed": self.seed,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "fold_accuracies": accs,
        }

        self.results.append(result)
        return result

    def save(self):
        # Save JSON
        with open(os.path.join(self.outdir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

        # Save CSV
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.outdir, "results.csv"), index=False)

        print(f"[SAVE] Results written to {self.outdir}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=ALLOWED_MODES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)

    exp = Experiment(cfg, mode=args.mode, seed=args.seed)

    classifiers = {
        "SVM": SVC(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=args.seed
        ),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=1000, random_state=args.seed),
        "DecisionTree": DecisionTreeClassifier(random_state=args.seed),
    }

    for name, clf in classifiers.items():
        exp.run(name, clf)

    exp.save()


if __name__ == "__main__":
    main()

