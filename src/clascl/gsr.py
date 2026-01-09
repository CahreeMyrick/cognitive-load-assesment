# src/clascl/gsr.py
from __future__ import annotations

import os
import glob
import warnings

import numpy as np
import pandas as pd
import neurokit2 as nk

from ..utils import ensure_dir, block_specific_filename


# ===============================
# Silence NeuroKit warnings
# ===============================
warnings.filterwarnings(
    "ignore",
    message=".*DFA_alpha2 related indices will not be calculated.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*invalid value encountered in scalar divide.*"
)


def extract_and_save_gsr_features(cfg: dict):
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

                if "gsr" not in df:
                    raise ValueError("Missing GSR column")

                gsr = df["gsr"].values.astype(float)

                # ----------------------------------
                # Clean GSR
                # ----------------------------------
                gsr_clean = nk.eda_clean(
                    gsr,
                    sampling_rate=sr,
                    method="BioSPPy"
                )

                # ----------------------------------
                # Phasic decomposition
                # ----------------------------------
                cvx = nk.eda_phasic(
                    gsr_clean,
                    sampling_rate=sr,
                    method="cvxEDA"
                )
                phasic = cvx["EDA_Phasic"].values

                mean_eda = float(np.mean(gsr_clean))

                # ----------------------------------
                # SCR peak detection
                # ----------------------------------
                _, eda_feats = nk.eda_peaks(
                    phasic,
                    sampling_rate=sr,
                    method="neurokit"
                )

                scr_amp = eda_feats.get("SCR_Amplitude", np.array([]))
                scr_rise = eda_feats.get("SCR_RiseTime", np.array([]))
                scr_rec = eda_feats.get("SCR_RecoveryTime", np.array([]))

                # ----------------------------------
                # Block-level aggregation
                # ----------------------------------
                feat = {
                    # Counts
                    "SCR_Count": int(len(scr_amp)),

                    # Amplitude
                    "SCR_Amplitude_Mean": float(np.mean(scr_amp)) if len(scr_amp) > 0 else 0.0,
                    "SCR_Amplitude_Max": float(np.max(scr_amp)) if len(scr_amp) > 0 else 0.0,

                    # Timing
                    "SCR_RiseTime_Mean": float(np.mean(scr_rise)) if len(scr_rise) > 0 else 0.0,
                    "SCR_RecoveryTime_Mean": float(np.mean(scr_rec)) if len(scr_rec) > 0 else 0.0,

                    # Tonic level
                    "GSR_Mean": mean_eda,
                }

                # Safety check
                if np.any(pd.isna(list(feat.values()))):
                    raise ValueError("NaN in GSR features")

                features.append(feat)
                labels.append(label_map[block])

            except Exception as e:
                print(
                    f"[GSR] Skip user {user_id}, block {block}: {e}"
                )

    # ----------------------------------
    # Save outputs
    # ----------------------------------
    features_df = pd.DataFrame(features).reset_index(drop=True)
    labels_df = pd.DataFrame(
        labels, columns=["Cognitive_Load_Label"]
    ).reset_index(drop=True)

    combined = pd.concat([features_df, labels_df], axis=1)

    features_df.to_csv(
        os.path.join(outdir, "GSRfeatures.csv"), index=False
    )
    labels_df.to_csv(
        os.path.join(outdir, "GSRlabels.csv"), index=False
    )
    combined.to_csv(
        os.path.join(outdir, "GSRfeatures_with_labels.csv"),
        index=False
    )

    print(
        f"[GSR] Saved {len(features_df)} samples, "
        f"{features_df.shape[1]} features"
    )

    return features_df, labels_df
