
from __future__ import annotations
import os, glob
import pandas as pd
import numpy as np
import neurokit2 as nk
from .utils import ensure_dir, block_specific_filename

def extract_and_save_gsr_features(cfg: dict):
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
                gsr = df["gsr"].values

                gsr_clean = nk.eda_clean(gsr, sampling_rate=sr, method="BioSPPy")
                cvx = nk.eda_phasic(gsr_clean, sampling_rate=sr, method="cvxEDA")
                p1 = cvx["EDA_Phasic"].values
                mean_eda = float(np.mean(gsr_clean))

                _, eda_feats = nk.eda_peaks(p1, sampling_rate=sr, method="neurokit")

                eda_features_df = pd.DataFrame({
                    "SCR_Onsets": eda_feats.get("SCR_Onsets", np.array([0])),
                    "SCR_Peaks": eda_feats.get("SCR_Peaks", np.array([0])),
                    "SCR_Height": eda_feats.get("SCR_Height", np.array([0.0])),
                    "SCR_Amplitude": eda_feats.get("SCR_Amplitude", np.array([0.0])),
                    "SCR_RiseTime": eda_feats.get("SCR_RiseTime", np.array([0.0])),
                    "SCR_Recovery": eda_feats.get("SCR_Recovery", np.array([0.0])),
                    "SCR_RecoveryTime": eda_feats.get("SCR_RecoveryTime", np.array([0.0])),
                    "mean_eda": np.array([mean_eda] * len(eda_feats.get("SCR_Onsets", [0]))),
                }).fillna(0.0)

                for _, feature_set in eda_features_df.iterrows():
                    features.append(feature_set.to_dict())
                    labels.append(label_map[block])

            except Exception as e:
                print(f"[GSR] Skip user {user_id}, block {block}: {e}")

    features_df = pd.DataFrame(features).reset_index(drop=True)
    labels_df = pd.DataFrame(labels, columns=["Cognitive_Load_Label"]).reset_index(drop=True)
    combined = pd.concat([features_df, labels_df], axis=1)

    features_df.to_csv(os.path.join(outdir, "GSRfeatures.csv"), index=False)
    labels_df.to_csv(os.path.join(outdir, "GSRlabels.csv"), index=False)
    combined.to_csv(os.path.join(outdir, "GSRfeatures_with_labels.csv"), index=False)

    return features_df, labels_df
