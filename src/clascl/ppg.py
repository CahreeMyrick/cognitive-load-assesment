# src/clascl/ppg.py
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd
import neurokit2 as nk

from src.clascl.utils import ensure_dir, block_specific_filename


# ===============================
#Silence  NeuroKit warnings
# ===============================
warnings.filterwarnings(
    "ignore",
    message=".*DFA_alpha2 related indices will not be calculated.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*invalid value encountered in scalar divide.*"
)


def extract_and_save_ppg_features(cfg: dict):
    block_details_dir = cfg["dataset"]["block_details_dir"]
    participants_dir = cfg["dataset"]["participants_dir"]
    blocks_of_interest = cfg["blocks_of_interest"]
    label_map = {int(k): v for k, v in cfg["cognitive_load_labels"].items()}
    sr = int(cfg.get("sampling_rate", 256))
    outdir = cfg.get("outputs_dir", "outputs")

    ensure_dir(outdir)

    block_details_paths = glob.glob(os.path.join(block_details_dir, "*.csv"))

    features = []
    labels = []

    for block_details_path in block_details_paths:
        block_details = pd.read_csv(block_details_path)

        filtered = (
            block_details[block_details["Block"].isin(blocks_of_interest)]
            .sort_values(by="Block")
        )

        user_id = os.path.basename(block_details_path).split("_")[0]

        for _, row in filtered.iterrows():
            block = int(row["Block"])
            ppg_file = str(row["EDA&PPG File"])
            fname = block_specific_filename(block, ppg_file)
            file_path = os.path.join(
                participants_dir, user_id, "by_block", fname
            )

            try:
                df = pd.read_csv(file_path)

                if "ppg" not in df:
                    raise ValueError("Missing PPG column")

                ppg = df["ppg"].values.astype(float)

                # -------------------------------
                # Clean + peak detection
                # -------------------------------
                ppg_signals, info = nk.ppg_process(
                    ppg,
                    sampling_rate=sr,
                    heart_rate_method="peak"
                )

                peaks = info.get("PPG_Peaks", None)
                if peaks is None or len(peaks) < 3:
                    raise ValueError("Insufficient PPG peaks")

                # -------------------------------
                # HRV (time-domain ONLY)
                # -------------------------------
                hrv = nk.hrv_time(
                    peaks,
                    sampling_rate=sr,
                    show=False
                )

                feat = {
                    "HRV_RMSSD": hrv["HRV_RMSSD"].iloc[0],
                    "HRV_MeanNN": hrv["HRV_MeanNN"].iloc[0],
                    "HRV_SDNN": hrv["HRV_SDNN"].iloc[0],
                    "PPG_Clean_Mean": ppg_signals["PPG_Clean"].mean(),
                }

                # Safety: skip NaN rows
                if np.any(pd.isna(list(feat.values()))):
                    raise ValueError("NaN in extracted features")

                features.append(feat)
                labels.append(label_map[block])

            except Exception as e:
                print(
                    f"[PPG] Skip user {user_id}, block {block}: {e}"
                )

    features_df = pd.DataFrame(features).reset_index(drop=True)
    labels_df = pd.DataFrame(labels, columns=["Cognitive_Load_Label"])

    combined = pd.concat([features_df, labels_df], axis=1)

    features_df.to_csv(
        os.path.join(outdir, "PPGfeatures.csv"), index=False
    )
    labels_df.to_csv(
        os.path.join(outdir, "PPGlabels.csv"), index=False
    )
    combined.to_csv(
        os.path.join(outdir, "PPGfeatures_with_labels.csv"), index=False
    )

    print(
        f"[PPG] Saved {len(features_df)} samples, "
        f"{features_df.shape[1]} features"
    )

    return features_df, labels_df
