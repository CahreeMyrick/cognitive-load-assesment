
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
from typing import Tuple, List
from .utils import ensure_dir

import EEGEXTRACT2 as eeg

FS = 256

def extract_user_id(path: str) -> str:
    return os.path.basename(path)

def extract_level_number(file_path: str) -> int:
    filename = os.path.basename(file_path)
    base = filename.replace(".csv", "")
    number_str = base.split("_")[-1]
    return int(number_str)

def extract_all_levels_files(user_dir: str) -> List[str]:
    files = glob.glob(os.path.join(user_dir, "*data_level*"))
    files.sort(key=extract_user_id)
    return files

def extract_all_user_eeg_dirs(root: str) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])

def normalize_to_custom_range(data: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

def segment_data_func(file: str, sfreq: int = 256, seg_seconds: int = 10, n_segments: int = 18):
    df = pd.read_csv(file)
    if df.isna().sum().sum() > 0:
        print(f"[EEG] NaNs in {file}, skipping.")
        return None

    seg_len = int(seg_seconds * sfreq)
    total_samples = len(df)
    if total_samples < n_segments * seg_len:
        print(f"[EEG] Not enough samples for {file}. Expected {n_segments*seg_len}, got {total_samples}.")
        return None

    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        seg = df.iloc[start:end].copy()
        seg.loc[:, "segment"] = i
        segments.append(seg)

    segmented_df = pd.concat(segments, ignore_index=True)
    channels = [c for c in segmented_df.columns if c not in ("segment", "Timestamp")]
    n_channels = len(channels)

    data = np.empty((n_channels, seg_len, len(segments)))
    for i, seg in enumerate(segments):
        for j, ch in enumerate(channels):
            data[j, :, i] = seg[ch].values
    return data, []

def aggregate_connectivity_features(connectivity_features: np.ndarray, epochs: int) -> np.ndarray:
    num_channels = 4
    num_pairs = num_channels * (num_channels - 1) // 2
    return np.mean(connectivity_features.reshape(-1, num_pairs, epochs), axis=1)

def feature_extraction(eegData: np.ndarray, bin_min, bin_max, binWidth) -> Tuple[np.ndarray, list]:
    n_channels, _, epochs = eegData.shape

    features = []
    feature_names = []

    ShannonRes = eeg.shannonEntropy(eegData, bin_min, bin_max, binWidth).T
    features.append(ShannonRes)
    feature_names += [f"ShannonEntropy_{i}" for i in range(ShannonRes.shape[1])]

    medianFreqRes = eeg.medianFreq(eegData, FS).T
    features.append(medianFreqRes)
    feature_names += [f"MedianFreq_{i}" for i in range(medianFreqRes.shape[1])]

    std_res = eeg.eegStd(eegData).T
    features.append(std_res)
    feature_names += [f"Std_{i}" for i in range(std_res.shape[1])]

    subbands = ["delta", "theta", "alpha", "beta", "gamma"]
    bands_dict = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 100)}
    for band in subbands:
        eeg_band = eeg.filt_data(eegData, *bands_dict[band], FS)
        sh_b = eeg.shannonEntropy(eeg_band, bin_min, bin_max, binWidth).T
        features.append(sh_b)
        feature_names += [f"ShannonEntropy_{band}_{i}" for i in range(sh_b.shape[1])]

    HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)
    features += [HjorthMob.T, HjorthComp.T]
    feature_names += [f"HjorthMob_{i}" for i in range(HjorthMob.shape[1])]
    feature_names += [f"HjorthComp_{i}" for i in range(HjorthComp.shape[1])]

    bands = ["alpha", "beta", "gamma"]
    bdict = {"alpha": (8, 12), "beta": (12, 30), "gamma": (30, 100)}
    for band in bands:
        bp = eeg.bandPower(eegData, *bdict[band], FS).T
        features.append(bp)
        feature_names += [f"BandPower_{band}_{i}" for i in range(bp.shape[1])]

    # MI
    mi_features = []
    for ch1 in range(n_channels):
        for ch2 in range(ch1 + 1, n_channels):
            mi = eeg.calculate2Chan_MI(eegData, ch1, ch2, bin_min, bin_max, binWidth)
            if mi.ndim == 1: mi = mi[:, None]
            mi_features.append(mi)
    if mi_features:
        mi_features = np.concatenate(mi_features, axis=1)
        mi_features = aggregate_connectivity_features(mi_features, epochs).T
        features.append(mi_features)
        feature_names += [f"MI_{i}" for i in range(mi_features.shape[1])]

    # PLI
    pli_feats = []
    for ch1 in range(n_channels):
        for ch2 in range(ch1 + 1, n_channels):
            pli = eeg.phaseLagIndex(eegData, ch1, ch2)
            if pli.ndim == 1: pli = pli[:, None]
            pli_feats.append(pli)
    if pli_feats:
        pli_feats = np.concatenate(pli_feats, axis=1)
        pli_feats = aggregate_connectivity_features(pli_feats, epochs).T
        features.append(pli_feats)
        feature_names += [f"PLI_{i}" for i in range(pli_feats.shape[1])]

    try:
        concatenated = np.concatenate(features, axis=1)
    except ValueError as e:
        for i, f in enumerate(features):
            print(f"[EEG] Feature {i} shape: {getattr(f, 'shape', None)}")
        raise
    return concatenated, feature_names

def extract_and_save_eeg_features(cfg: dict):
    eeg_root = cfg["dataset"]["eeg_root"]
    labels_root = cfg["dataset"]["labels_root"]
    outdir = cfg.get("outputs_dir", "outputs")
    ensure_dir(outdir)

    features_list = []
    labels_list = []

    user_dirs = extract_all_user_eeg_dirs(eeg_root)
    for udir in user_dirs:
        user_id = extract_user_id(udir)
        labels_df = pd.read_csv(os.path.join(labels_root, f"{user_id}.csv"))

        level_files = extract_all_levels_files(udir)
        for fpath in level_files:
            level = extract_level_number(fpath)
            result = segment_data_func(fpath)
            if result is None:
                continue

            # Label mapping per segment
            labels_df[f"lvl_{level}"] = labels_df[f"lvl_{level}"].apply(
                lambda x: "low" if 1 <= x <= 3 else ("medium" if 4 <= x <= 6 else "high")
            )
            level_labels = labels_df[f"lvl_{level}"].values

            segmented_data, _ = result
            data_min, data_max = np.min(segmented_data), np.max(segmented_data)
            binWidth = 1
            bin_min, bin_max = data_min, data_max + binWidth
            normalized = normalize_to_custom_range(segmented_data, bin_min, bin_max)

            feats, feat_names = feature_extraction(normalized, bin_min, bin_max, binWidth)
            features_list.append(feats)
            labels_list.append(level_labels)

    if not features_list:
        raise RuntimeError("[EEG] No features extracted; check your paths/data.")

    all_features = np.vstack(features_list)
    all_labels = np.hstack(labels_list)

    feats_df = pd.DataFrame(all_features, columns=feat_names).reset_index(drop=True)
    labels_df = pd.DataFrame(all_labels, columns=["label"]).reset_index(drop=True)
    labels_df["label"] = labels_df["label"].map({"low": 0, "medium": 1, "high": 2})

    feats_df.to_csv(os.path.join(outdir, "EEGfeatures.csv"), index=False)
    labels_df.to_csv(os.path.join(outdir, "EEGlabels.csv"), index=False)
    pd.concat([feats_df, labels_df], axis=1).to_csv(os.path.join(outdir, "EEGfeatures_with_labels.csv"), index=False)

    return feats_df, labels_df
