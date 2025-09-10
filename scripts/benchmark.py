
import argparse, os
import pandas as pd
from clascl.models import train_and_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X = pd.read_csv(args.features)
    y = pd.read_csv(args.labels).squeeze()
    reports = train_and_eval(X, y, test_size=args.test_size, seed=args.seed)
    for name, r in reports.items():
        print(f"\n=== {name} ===")
        print("Accuracy:", r["accuracy"])
        print(r["report"])

if __name__ == "__main__":
    main()
