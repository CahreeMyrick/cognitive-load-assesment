
import argparse
from src.clascl.utils import load_config
from src.clascl.ppg import extract_and_save_ppg_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    extract_and_save_ppg_features(cfg)
    print("[PPG] Done. Files saved in:", cfg.get("outputs_dir", "outputs"))

if __name__ == "__main__":
    main()
