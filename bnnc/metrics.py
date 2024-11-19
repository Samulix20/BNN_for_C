from math import log

import numpy as np
import numpy.typing as npt

import pandas as pd

def global_uncertainty(p: npt.NDArray) -> float:
    avg = np.mean(p, axis=0)
    h = -np.sum(avg * np.log(avg, where=(avg > 0)))
    return h


def normalized_global_uncertainty(p: npt.NDArray) -> float:
    _, num_classes = p.shape
    return global_uncertainty(p) / np.log(num_classes)


def expexted_entropy(p: npt.NDArray) -> float:
    return np.mean(-np.sum(p * np.log(p, where=(p > 0)), axis=1))


def class_predicted(p: npt.NDArray) -> int:
    return np.mean(p, axis=0).argmax()


def confidence(p: npt.NDArray):
    return np.mean(p, axis=0).max()

def accurate(p, target):
    return 1 if class_predicted(p) == target else 0


def accuracy(data) -> float:
    return data["accurate"].to_numpy().mean()


def expected_calibration_error(data, nbins=10):
    bsize = 1 / nbins
    bins = np.arange(0, 1 + bsize, bsize)

    nsamples = len(data)
    ece = 0

    confs = data["confidence"].to_numpy()
    accs = data["accurate"].to_numpy()

    for i in range(nbins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        bin_accs = accs[mask]
        bin_confs = confs[mask]

        if len(bin_accs) > 0:
            acc = np.mean(bin_accs)
            conf = np.mean(bin_confs)
            ece += len(bin_accs) * abs(acc - conf) / nsamples
    
    return ece


def uncertainty_calibration_error(data, nbins=10):
    bsize = 1 / nbins
    bins = np.arange(0, 1 + bsize, bsize)

    nsamples = len(data)
    uce = 0

    confs = data["confidence"].to_numpy()
    accs = data["accurate"].to_numpy()
    uncs = data["uncertainty"].to_numpy()

    for i in range(nbins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        bin_accs = accs[mask]
        bin_uncs = uncs[mask]

        if len(bin_accs) > 0:
            err = 1 - np.mean(bin_accs)
            unc = np.mean(bin_uncs)
            uce += len(bin_accs) * abs(err - unc) / nsamples
    
    return uce


# Derived from the reliability plot
def reliability_error(data, nbins=10):

    targets = data["target"].to_numpy()
    averages = np.stack(data["average"])
    _, num_classes = averages.shape

    labels_one_hot = np.zeros((len(targets), num_classes))
    labels_one_hot[range(len(targets)), targets] = 1

    # Bins
    bsize = 1 / nbins
    p_groups = np.arange(0, 1 + bsize, bsize)
    centers = p_groups[:-1] + (p_groups[1:] - p_groups[:-1]) / 2

    ere = 0

    for i in range(len(p_groups) - 1):
        p_min = p_groups[i]
        p_max = p_groups[i + 1]
        group = labels_one_hot[(averages >= p_min) & (averages < p_max)]
        if len(group) > 0:
            ere += abs(np.mean(group) - centers[i]) / len(centers)

    return ere


def analyze_predictions(predictions, targets):
    l = []
    for prediction, target in zip(predictions, targets):
        l.append({
            "prediction": prediction,
            "average": prediction.mean(axis=0),
            "confidence": confidence(prediction),
            "uncertainty": normalized_global_uncertainty(prediction),
            "class_predicted": class_predicted(prediction),
            "target": target,
            "accurate": accurate(prediction, target)
        })
    data = pd.DataFrame(l)

    return {
        "analyzed_predictions": data,
        "ece": expected_calibration_error(data),
        "uce": uncertainty_calibration_error(data),
        "re": reliability_error(data),
        "acc": accuracy(data)
    }

def match_ratio(a: npt.NDArray, b: npt.NDArray) -> float:

    match_mask = a["class_predicted"].to_numpy() == b["class_predicted"].to_numpy() 
    diff_mask = ~match_mask

    return np.mean(match_mask), diff_mask

def uncertainty_quality(data, nths = 50):
    nsamples = len(data)

    mask_correct = data["accurate"].to_numpy() == 1
    mask_error = ~mask_correct

    uncs = data["uncertainty"].to_numpy()
    correct_uncs = uncs[mask_correct]
    error_uncs = uncs[mask_error]

    step_size = 1 / nths
    uths = np.arange(step_size, 1 + step_size, step_size)

    r = []

    for i, uth in enumerate(uths):
        nau = (correct_uncs > uth).sum()
        nac = (correct_uncs <= uth).sum()
        nic = (error_uncs <= uth).sum()
        niu = (error_uncs > uth).sum()

        p_acc_cert = nac / (nac + nic)
        p_unc_inn = niu / (nic + niu)

        r.append({
            "uth": uth,
            "p_acc_cert": p_acc_cert,
            "p_unc_inn": p_unc_inn,
            "p_cert": (nac + nic) / nsamples
        })

    return pd.DataFrame(r)

def reliability_data(data, nbins = 10):
    targets = data["target"].to_numpy()
    averages = np.stack(data["average"])
    _, num_classes = averages.shape

    labels_one_hot = np.zeros((len(targets), num_classes))
    labels_one_hot[range(len(targets)), targets] = 1

    # Bins
    bsize = 1 / nbins
    p_groups = np.arange(0, 1 + bsize, bsize)
    centers = p_groups[:-1] + (p_groups[1:] - p_groups[:-1]) / 2

    r = []

    for i in range(len(p_groups) - 1):
        p_min = p_groups[i]
        p_max = p_groups[i + 1]
        group = labels_one_hot[(averages >= p_min) & (averages < p_max)]
        
        if len(group) > 0:
            r.append(group.mean())
        else:
            r.append(0)

    return {
        "predicted": centers,
        "observed": r,
    }
