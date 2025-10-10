# scripts/run_fusion.py

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.clascl.utils import load_config, ensure_dir, downsample_dataframe


# ============================================================
# FusionDataset
# ============================================================

class FusionDataset:
    def __init__(
        self,
        cfg: dict,
        use_eeg: bool = False,
        use_gsr: bool = False,
        use_ppg: bool = False,
    ):
        assert use_eeg or use_gsr or use_ppg, "Select at least one modality"

        self.cfg = cfg
        self.outdir = cfg.get("outputs_dir", "outputs")

        self.use_eeg = use_eeg
        self.use_gsr = use_gsr
        self.use_ppg = use_ppg

        self.X = None
        self.y = None

        self._load_and_fuse()

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    def _load_modality(self, name):
        feat = pd.read_csv(os.path.join(self.outdir, f"{name}features.csv"))
        lab = pd.read_csv(os.path.join(self.outdir, f"{name}labels.csv")).squeeze()
        print(f"[LOAD] {name}: X={feat.shape}, y={lab.shape}")
        return feat, lab

    # --------------------------------------------------------
    # Core fusion logic
    # --------------------------------------------------------

    def _load_and_fuse(self):
        X_parts = []
        y_ref = None

        # Determine target length = minimum samples across selected modalities
        lengths = []

        if self.use_eeg:
            X_eeg, y_eeg = self._load_modality("EEG")
            lengths.append(len(X_eeg))
        if self.use_gsr:
            X_gsr, y_gsr = self._load_modality("GSR")
            lengths.append(len(X_gsr))
        if self.use_ppg:
            X_ppg, y_ppg = self._load_modality("PPG")
            lengths.append(len(X_ppg))

        target_len = min(lengths)
        print(f"[FUSION] Target aligned length = {target_len}")

        # Load + downsample + concatenate
        if self.use_eeg:
            X_eeg, y_eeg = self._load_modality("EEG")
            X_eeg = downsample_dataframe(X_eeg, target_len)
            X_parts.append(X_eeg)
            y_ref = y_eeg.iloc[:target_len]

        if self.use_gsr:
            X_gsr, y_gsr = self._load_modality("GSR")
            X_gsr = downsample_dataframe(X_gsr, target_len)
            X_parts.append(X_gsr)
            if y_ref is None:
                y_ref = y_gsr.iloc[:target_len]

        if self.use_ppg:
            X_ppg, y_ppg = self._load_modality("PPG")
            X_ppg = downsample_dataframe(X_ppg, target_len)
            X_parts.append(X_ppg)
            if y_ref is None:
                y_ref = y_ppg.iloc[:target_len]

        X = pd.concat(X_parts, axis=1)
        y = y_ref.reset_index(drop=True)

        # ----------------------------------------------------
        # Impute + Scale — done before CV split (no leakage)
        # ----------------------------------------------------
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        self.X = X
        self.y = y

        print(f"[FUSION] Final X={X.shape}, y={y.shape}")

    # --------------------------------------------------------
    # CV evaluation
    # --------------------------------------------------------

    def evaluate(self, clf, name, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []

        print(f"\n[CV] {name}")
        for fold, (tr, te) in enumerate(skf.split(self.X, self.y), 1):
            Xtr, Xte = self.X.iloc[tr], self.X.iloc[te]
            ytr, yte = self.y.iloc[tr], self.y.iloc[te]

            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xte)

            acc = accuracy_score(yte, ypred)
            accs.append(acc)

            print(f"  Fold {fold}: acc={acc:.4f}")

        print(
            f"{name}: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}"
        )
        return accs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_eeg", action="store_true")
    ap.add_argument("--use_gsr", action="store_true")
    ap.add_argument("--use_ppg", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg.get("outputs_dir", "outputs"))

    dataset = FusionDataset(
        cfg,
        use_eeg=args.use_eeg,
        use_gsr=args.use_gsr,
        use_ppg=args.use_ppg,
    )

    classifiers = {
        "SVM": SVC(),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=500, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    for name, clf in classifiers.items():
        dataset.evaluate(clf, name)


if __name__ == "__main__":
    main()
