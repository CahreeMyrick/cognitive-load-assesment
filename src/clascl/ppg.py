
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import neurokit2 as nk
from .utils import ensure_dir, block_specific_filename

def extract_and_save_ppg_features(cfg: dict):
    block_details_dir = cfg["dataset"]["block_details_dir"]
    participants_dir = cfg["dataset"]["participants_dir"]
    blocks_of_interest = cfg["blocks_of_interest"]
    label_map = {int(k): v for k, v in cfg["cognitive_load_labels"].items()}
    sr = int(cfg.get("sampling_rate", 256))
    outdir = cfg.get("outputs_dir", "outputs")
    ensure_dir(outdir)

    all_users_block_details_paths = glob.glob(os.path.join(block_details_dir, "*.csv"))
    features = []
    labels = []

    for idx, block_details_path in enumerate(all_users_block_details_paths, 1):
        block_details = pd.read_csv(block_details_path)
        filtered = block_details[block_details["Block"].isin(blocks_of_interest)]
        filtered = filtered.sort_values(by="Block")
        user_id = os.path.basename(block_details_path).split("_")[0]

        for _, row in filtered.iterrows():
            block = int(row["Block"])
            ppg_file = str(row["EDA&PPG File"])
            fname = block_specific_filename(block, ppg_file)
            file_path = os.path.join(participants_dir, user_id, "by_block", fname)

            try:
                df = pd.read_csv(file_path)
                ppg = df["ppg"].values.flatten()
                ppg_signals, info = nk.ppg_process(ppg, sampling_rate=sr)
                analysis = nk.ppg_analyze(ppg_signals, sampling_rate=sr, method="interval-related")

                feat = {
                    "HRV_RMSSD": float(analysis["HRV_RMSSD"]),
                    "HRV_MeanNN": float(analysis["HRV_MeanNN"]),
                    "HRV_SDNN": float(analysis["HRV_SDNN"]),
                    "HRV_ApEn": float(analysis["HRV_ApEn"]),
                    "PPG_Clean_Mean": float(ppg_signals["PPG_Clean"].mean()),
                }
                features.append(feat)
                labels.append(label_map[block])

            except Exception as e:
                print(f"[PPG] Skip user {user_id}, block {block}: {e}")

    features_df = pd.DataFrame(features).reset_index(drop=True)
    labels_df = pd.DataFrame(labels, columns=["Cognitive_Load_Label"]).reset_index(drop=True)
    combined = pd.concat([features_df, labels_df], axis=1)

    features_df.to_csv(os.path.join(outdir, "PPGfeatures.csv"), index=False)
    labels_df.to_csv(os.path.join(outdir, "PPGlabels.csv"), index=False)
    combined.to_csv(os.path.join(outdir, "PPGfeatures_with_labels.csv"), index=False)

    return features_df, labels_df
