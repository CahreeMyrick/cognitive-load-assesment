
import argparse
from clascl.utils import load_config
from clascl.eeg import extract_and_save_eeg_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    extract_and_save_eeg_features(cfg)
    print("[EEG] Done. Files saved in:", cfg.get("outputs_dir", "outputs"))

if __name__ == "__main__":
    main()
