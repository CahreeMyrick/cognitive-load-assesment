from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd

from .utils import ensure_dir
from . import eegextract as eeg

FS = 256


def extract_user_id(path: str) -> str:
    return os.path.basename(path)


def extract_level_number(file_path: str) -> int:
    fname = os.path.basename(file_path)
    return int(fname.replace(".csv", "").split("_")[-1])


def extract_all_levels_files(user_dir: str):
    files = glob.glob(os.path.join(user_dir, "*data_level*"))
    files.sort(key=extract_level_number)
    return files


def extract_all_user_eeg_dirs(root: str):
    return sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))


def normalize_to_custom_range(data: np.ndarray, new_min: float, new_max: float):
    mn = np.min(data)
    mx = np.max(data)
    if mx == mn:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn) * (new_max - new_min) + new_min


def segment_data_func(
    file: str,
    sfreq: int = FS,
    seg_seconds: int = 10,
    n_segments: int = 18,
):
    df = pd.read_csv(file)

    # 🔧 Interpolate NaNs first
    if df.isna().any().any():
        df = df.interpolate(method="linear", limit=50)
        if df.isna().any().any():
            print(f"[EEG] Unrecoverable NaNs in {file}, skipping.")
            return None

    seg_len = int(seg_seconds * sfreq)
    total = len(df)

    if total < seg_len * n_segments:
        print(f"[EEG] Not enough samples in {file}")
        return None

    channels = [c for c in df.columns if c not in ("Timestamp", "segment")]
    data = np.empty((len(channels), seg_len, n_segments))

    for i in range(n_segments):
        seg = df.iloc[i * seg_len : (i + 1) * seg_len]
        for j, ch in enumerate(channels):
            data[j, :, i] = seg[ch].values

    return data


def extract_and_save_eeg_features(cfg: dict):
    eeg_root = cfg["dataset"]["eeg_root"]
    labels_root = cfg["dataset"]["labels_root"]
    outdir = cfg.get("outputs_dir", "outputs")

    ensure_dir(outdir)

    X_all = []
    y_all = []
    feature_names = None

    for udir in extract_all_user_eeg_dirs(eeg_root):
        user_id = extract_user_id(udir)
        label_path = os.path.join(labels_root, f"{user_id}.csv")

        if not os.path.exists(label_path):
            print(f"[EEG] Missing labels for {user_id}, skipping.")
            continue

        labels_df = pd.read_csv(label_path)

        for fpath in extract_all_levels_files(udir):
            level = extract_level_number(fpath)
            segmented = segment_data_func(fpath)
            if segmented is None:
                continue

            lvl_col = f"lvl_{level}"
            if lvl_col not in labels_df:
                continue

            y = labels_df[lvl_col].map(
                lambda x: 0 if 1 <= x <= 3 else (1 if 4 <= x <= 6 else 2)
            ).values

            bin_min = np.min(segmented)
            bin_max = np.max(segmented) + 1

            norm = normalize_to_custom_range(segmented, bin_min, bin_max)

            X, curr_names = eeg.feature_extraction(
                norm, bin_min, bin_max, 1
            )

            if feature_names is None:
                feature_names = curr_names

            if X.shape[1] != len(feature_names):
                print(
                    f"[EEG] Feature mismatch in {fpath} "
                    f"(got {X.shape[1]}, expected {len(feature_names)}), skipping."
                )
                continue

            X_all.append(X)
            y_all.append(y)

    if not X_all:
        raise RuntimeError("[EEG] No EEG features extracted")

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    pd.DataFrame(X_all, columns=feature_names).to_csv(
        os.path.join(outdir, "EEGfeatures.csv"), index=False
    )
    pd.DataFrame(y_all, columns=["label"]).to_csv(
        os.path.join(outdir, "EEGlabels.csv"), index=False
    )

    print(
        f"[EEG] Saved {X_all.shape[0]} samples, "
        f"{X_all.shape[1]} features"
    )

    return X_all, y_all
