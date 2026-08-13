"""Custom scoring utilities for Raman model selection.

This module defines scorers compatible with scikit-learn search APIs,
including an IS-based spectral quality score and patient-level majority-vote
accuracy.
"""

from collections import Counter

import numpy as np
from IS_Score.IS_Score import getIS_Score
from sklearn.metrics import accuracy_score
from ramantune.utils import BASELINE_STR, DENOISING_STR, TO_SPECTRA_STR, TO_VALUES_STR

def is_score(estimator, X, y):
    """
    Compute mean IS score across transformed spectra.

    Parameters
    ----------
    estimator : object
        Fitted pipeline estimator exposing ``named_steps`` with conversion,
        baseline, and denoising transformers.
    X : pandas.DataFrame
        Input spectra matrix.
    y : array-like
        Target labels. Unused, included for scorer API compatibility.

    Returns
    -------
    float
        Mean IS score over all samples.
    """
    to_spectra, to_values = estimator.named_steps[TO_SPECTRA_STR], estimator.named_steps[TO_VALUES_STR]
    baseline_transf, denoising_transf = estimator.named_steps[BASELINE_STR], estimator.named_steps[DENOISING_STR]

    baseline_corr = baseline_transf.transform(to_spectra.transform(X))
    smoothed = denoising_transf.transform(to_spectra.transform(X))

    baseline_corr_val = to_values.transform(baseline_corr)
    smoothed_val = to_values.transform(smoothed)

    scores = []

    for i in range(len(smoothed_val)):
        score_i = getIS_Score(raw_sp=smoothed_val.iloc[i],
                              baseline_corrected_sp=baseline_corr_val.iloc[i],
                              sp_axis=smoothed_val.columns)
        scores.append(score_i)

    return float(np.mean(scores))

def patient_accuracy(estimator, X, y, **kwargs):
    """
    Compute patient-level accuracy via majority voting.

    Parameters
    ----------
    estimator : object
        Fitted estimator implementing ``predict``.
    X : pandas.DataFrame
        Input samples used for prediction.
    y : array-like
        Sample-level true labels.
    **kwargs : dict
        Additional scorer arguments. Must include ``groups`` containing
        patient identifiers indexed like ``X``.

    Returns
    -------
    float
        Accuracy score computed at patient level.

    Notes
    -----
    For each patient group, predicted class is the majority class among that
    patient's samples and true class is taken from the first sample label.
    """
    groups = kwargs['groups'][X.index.values]

    y_pred = estimator.predict(X)
    patients = {}
    for pred, true, group in zip(y_pred, y, groups):
        if group not in patients:
            patients[group] = {"preds": [], "true": []}
        patients[group]["preds"].append(pred)
        patients[group]["true"].append(true)

    patient_true = []
    patient_pred = []
    for group, values in patients.items():
        majority_pred = Counter(values["preds"]).most_common(1)[0][0]
        # Assumes all samples of a patient share the same true label
        patient_true.append(values["true"][0])
        patient_pred.append(majority_pred)

    return accuracy_score(patient_true, patient_pred)
