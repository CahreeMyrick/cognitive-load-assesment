# run_pipeline.py
import argparse
from src.utils import load_config

from src.cl_drive.eeg import extract_and_save_eeg_features
from src.clascl.gsr import extract_and_save_gsr_features
from src.clascl.ppg import extract_and_save_ppg_features

MODALITY_DISPATCH = {
    "eeg": extract_and_save_eeg_features,
    "gsr": extract_and_save_gsr_features,
    "ppg": extract_and_save_ppg_features,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITY_DISPATCH.keys(),
        required=True,
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    for modality in args.modalities:
        print(f"[FEATURE EXTRACTION] Running {modality.upper()}")
        MODALITY_DISPATCH[modality](cfg)

    print("[FEATURE EXTRACTION] Done.")

if __name__ == "__main__":
    main()
