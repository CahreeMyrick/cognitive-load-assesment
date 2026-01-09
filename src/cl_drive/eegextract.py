

import numpy as np
from scipy import signal
from scipy import signal as ss
from sklearn.metrics import mutual_info_score
import statsmodels.tsa.api as tsa

# =========================================================
# GLOBAL CONSTANTS
# =========================================================
FS = 256   # Sampling frequency (Hz)

############################################
# Filtering
############################################

def filt_data(eegData, lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.lfilter(b, a, eegData, axis=1)


############################################
# Entropy / Complexity
############################################

def shannonEntropy(eegData, bin_min, bin_max, binWidth):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    bins = np.arange(bin_min, bin_max + binWidth, binWidth)

    for ch in range(eegData.shape[0]):
        for ep in range(eegData.shape[2]):
            counts, _ = np.histogram(eegData[ch, :, ep], bins=bins)
            p = counts[counts > 0]
            p = p / np.sum(p)
            H[ch, ep] = -np.sum(p * np.log2(p))

    return H


def hjorthParameters(xV):
    dx = np.diff(xV, axis=1)
    ddx = np.diff(dx, axis=1)

    var_x = np.mean(xV ** 2, axis=1)
    var_dx = np.mean(dx ** 2, axis=1)
    var_ddx = np.mean(ddx ** 2, axis=1)

    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt((var_ddx / var_dx) / (var_dx / var_x))

    return mobility, complexity


def eegStd(eegData):
    return np.std(eegData, axis=1)


############################################
# Spectral
############################################

def medianFreq(eegData, fs):
    out = np.zeros((eegData.shape[0], eegData.shape[2]))

    for ch in range(eegData.shape[0]):
        freqs, psd = signal.periodogram(eegData[ch, :, :], fs, axis=0)
        cumsum = np.cumsum(psd, axis=0)
        half = cumsum[-1] / 2
        out[ch] = freqs[np.argmax(cumsum >= half, axis=0)]

    return out


def bandPower(eegData, lowcut, highcut, fs):
    eeg_band = filt_data(eegData, lowcut, highcut, fs)
    _, psd = signal.periodogram(eeg_band, fs, axis=1)
    return np.mean(psd, axis=1)


############################################
# Connectivity
############################################

def calculate2Chan_MI(eegData, ch1, ch2, bin_min, bin_max, binWidth):
    epochs = eegData.shape[2]
    bins = np.arange(bin_min, bin_max + binWidth, binWidth)
    out = np.zeros(epochs)

    for ep in range(epochs):
        x = eegData[ch1, :, ep]
        y = eegData[ch2, :, ep]
        c_xy, _, _ = np.histogram2d(x, y, bins=bins)
        out[ep] = mutual_info_score(None, None, contingency=c_xy)

    return out


def phaseLagIndex(eegData, ch1, ch2):
    h1 = ss.hilbert(eegData[ch1, :, :])
    h2 = ss.hilbert(eegData[ch2, :, :])
    phase_diff = np.angle(h2) - np.angle(h1)
    return np.abs(np.mean(np.sign(phase_diff), axis=0))


############################################
# Feature Extraction
############################################

def feature_extraction(eegData, bin_min, bin_max, binWidth):
    n_channels, _, epochs = eegData.shape

    features = []
    feature_names = []

    def add(feat, prefix):
        features.append(feat)
        feature_names.extend([f"{prefix}_{i}" for i in range(feat.shape[1])])

    # Shannon Entropy
    H = shannonEntropy(eegData, bin_min, bin_max, binWidth).T
    add(H, "Shannon")

    # Median Frequency
    MF = medianFreq(eegData, FS).T
    add(MF, "MedianFreq")

    # Standard Deviation
    STD = eegStd(eegData).T
    add(STD, "Std")

    # Hjorth Parameters
    mob, comp = hjorthParameters(eegData)
    add(mob.T, "HjorthMob")
    add(comp.T, "HjorthComp")

    # Band Power
    for band, (lo, hi) in {
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 100),
    }.items():
        bp = bandPower(eegData, lo, hi, FS).T
        add(bp, f"BandPower_{band}")

    # Mutual Information (4 channels -> 6 pairs)
    mi_feats = []
    for ch1 in range(n_channels):
        for ch2 in range(ch1 + 1, n_channels):
            mi = calculate2Chan_MI(eegData, ch1, ch2, bin_min, bin_max, binWidth)
            mi_feats.append(mi[:, None])

    mi_feats = np.concatenate(mi_feats, axis=1)
    add(mi_feats, "MI")

    # Phase Lag Index
    pli_feats = []
    for ch1 in range(n_channels):
        for ch2 in range(ch1 + 1, n_channels):
            pli = phaseLagIndex(eegData, ch1, ch2)
            pli_feats.append(pli[:, None])

    pli_feats = np.concatenate(pli_feats, axis=1)
    add(pli_feats, "PLI")

    X = np.concatenate(features, axis=1)

    # Hard safety invariant
    assert X.shape[1] == len(feature_names), (
        f"Feature mismatch: X={X.shape[1]} names={len(feature_names)}"
    )

    return X, feature_names

