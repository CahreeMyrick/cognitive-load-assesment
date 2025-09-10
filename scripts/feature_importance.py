
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None, help="PNG path to save the bar chart")
    ap.add_argument("--title", default="Feature Importances (Random Forest)")
    args = ap.parse_args()

    X = pd.read_csv(args.features)
    y = pd.read_csv(args.labels).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=1200)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("RF test score:", score)

    std = np.std([t.feature_importances_ for t in clf.estimators_], axis=0)
    importances = pd.Series(clf.feature_importances_, index=X_train.columns)

    ax = importances.plot(kind="bar", yerr=std)
    ax.set_title(args.title)
    ax.set_ylabel("Mean decrease in impurity")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print("Saved:", args.out)
    else:
        plt.show()

if __name__ == "__main__":
    main()
