
import argparse, os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from clascl.utils import load_config, ensure_dir, downsample_dataframe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ppg_features", default=None, help="Override path; else uses outputs/PPGfeatures.csv")
    ap.add_argument("--ppg_labels", default=None, help="Override path; else uses outputs/PPGlabels.csv")
    ap.add_argument("--gsr_features", default=None, help="Override path; else uses outputs/GSRfeatures.csv")
    ap.add_argument("--gsr_labels", default=None, help="Override path; else uses outputs/GSRlabels.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = cfg.get("outputs_dir", "outputs")
    ensure_dir(outdir)

    ppg_feat = args.ppg_features or os.path.join(outdir, "PPGfeatures.csv")
    ppg_lab  = args.ppg_labels   or os.path.join(outdir, "PPGlabels.csv")
    gsr_feat = args.gsr_features or os.path.join(outdir, "GSRfeatures.csv")
    gsr_lab  = args.gsr_labels   or os.path.join(outdir, "GSRlabels.csv")

    X_ppg = pd.read_csv(ppg_feat)
    y_ppg = pd.read_csv(ppg_lab).squeeze()
    X_gsr = pd.read_csv(gsr_feat)
    y_gsr = pd.read_csv(gsr_lab).squeeze()

    # Downsample GSR to match PPG sample count
    X_gsr_ds = downsample_dataframe(X_gsr, len(X_ppg))

    combined = pd.concat([X_ppg, X_gsr_ds], axis=1)
    labels = y_ppg  # as in the notebook

    # Impute + Scale
    imputer = SimpleImputer(strategy="mean")
    combined = pd.DataFrame(imputer.fit_transform(combined), columns=combined.columns)

    scaler = StandardScaler()
    combined = pd.DataFrame(scaler.fit_transform(combined), columns=combined.columns)

    X_train, X_test, y_train, y_test = train_test_split(combined, labels, test_size=0.3, random_state=42)

    classifiers = {
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=1200),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Multi-Layer Perceptron': MLPClassifier(max_iter=500, random_state=1200),
        'Decision Tree': DecisionTreeClassifier(random_state=1200)
    }

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"{name} Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
