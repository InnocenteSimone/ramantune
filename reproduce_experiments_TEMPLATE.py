import os
import time
import numpy as np
import pandas as pd
import ramanspy as rp
import ramanspy.preprocessing as rpr

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
from ramantune.pipeline.raman_pipeline import RamanPipeline
from ramantune.search.search_space import DenoiserSpace, BaselineSpace, NormalizerSpace, ClassifierSpace, FeatureSelectionSpace
from ramantune.utils.config import DENOISING_STR, BASELINE_STR, NORMALIZE_STR, FEATURE_SELECTION_STR, CLASSIFIER_STR
from ramantune.search.strategies import GridSearchStrategy
from ramantune.search import RamanSearch

def ovarian_create_df(filepath, patient_counter = 1):
    FILEPATH_OVARIAN_HEALTHY_FIRST = Path("plasma_HC_1_10sec_5lp_2acc3 (1).txt") # Path to the first healthy ovarian Raman data file
    first = np.loadtxt(FILEPATH_OVARIAN_HEALTHY_FIRST)
    frequencies = [str(el) for el in pd.DataFrame(first)[0].values]
    intensities = []

    previous_patient = None
    patient = []

    for root, dirs, files in os.walk(filepath):
        for f in files:
            path = filepath / f
            tmp = pd.DataFrame(np.loadtxt(path))
            if tmp.shape[0] > 1480:
                print("Higher number of frequencies")
                tmp = tmp.iloc[:1480]

            if previous_patient is not None and int(f.split("_")[2]) != previous_patient:
                patient_counter += 1

            patient.append(patient_counter)
            previous_patient = int(f.split("_")[2])
            intensities.append(tmp[1].values)

    df = pd.DataFrame(intensities, columns=frequencies)
    df = df[df.columns.values[::-1]]
    df['patient'] = [str(el) for el in patient]
    return df, patient_counter

def format_ovarian_dataset():
    FILEPATH_HEALTHY = "..." # Path to healthy folder
    FILEPATH_OVARIAN = "..." # Path to ovarian cancer folder

    df_h, patients = ovarian_create_df(FILEPATH_HEALTHY, patient_counter=1)
    df_h['label'] = "HC"
    df_o, _ = ovarian_create_df(FILEPATH_OVARIAN, patient_counter=patients + 1)
    df_o['label'] = "OC"

    df = pd.DataFrame()
    df = pd.concat([df_h, df_o])

    return df

def format_covid_dataset():
    FILEPATH_COVID_WAVENUMBERS = "wave_number.txt" # Path to wavenumber raman data
    FILEPATH_RAW_COVID = "raw_COVID.txt" # Path to raw COVID-19 Raman data
    FILEPATH_RAW_SUSPECTED = "raw_Suspected.txt" # Path to raw Suspected Raman data
    FILEPATH_RAW_HEALTHY = "raw_Helthy.txt" # Path to raw Healthy Raman data
    FILEPATH_RAW_TUBE = "raw_Tube.txt" # Path to raw Tube Raman data

    wn = np.loadtxt(FILEPATH_COVID_WAVENUMBERS)

    raw_C = pd.DataFrame(np.loadtxt(FILEPATH_RAW_COVID).T, columns=wn)
    raw_C['label'] = "COVID"

    raw_S = pd.DataFrame(np.loadtxt(FILEPATH_RAW_SUSPECTED).T, columns=wn)
    raw_S['label'] = "SUSPECTED"

    raw_H = pd.DataFrame(np.loadtxt(FILEPATH_RAW_HEALTHY).T, columns=wn)
    raw_H['label'] = "HEALTHY"

    raw_T = pd.DataFrame(np.loadtxt(FILEPATH_RAW_TUBE).T, columns=wn)
    raw_T['label'] = "TUBE"

    df = pd.DataFrame()
    df = pd.concat([raw_C, raw_S, raw_H, raw_T])
    df = df.drop(columns=[400.0, 2112.0])

    return df



def apply_preprocessing(row, pip):
    return pip.apply(row)

def create_raman_spectrum(row):
    values = row.values.astype("float")
    frequencies = row.index.astype("float")
    return rp.Spectrum(values, frequencies)

def clean_dataframe(df, pipeline, return_dataframe=False):
  df['Spectra'] = df.apply(create_raman_spectrum, axis=1)
  df['Spectra_preprocessed'] = df['Spectra'].apply(apply_preprocessing, pip=pipeline)

  X_preprocessed = pd.DataFrame(df['Spectra_preprocessed'].apply(lambda x: dict(zip(x.spectral_axis, x.spectral_data))).tolist())

  if return_dataframe:
    return X_preprocessed
  return X_preprocessed.values

def setup_param_grid():
  denoiser_list = [
        DenoiserSpace("savgol", {"window_length": [7, 11], "polyorder": [3]}),
  ]

  baseline_list = [
      BaselineSpace("modpoly", {"poly_order": [4]}),
      BaselineSpace("bubblefill", {"min_bubble_widths": [100]}),
      BaselineSpace("iasls", {"lam": [100], "p": [0.001]}),
      BaselineSpace("asls", {"lam": [100]}),
  ]

  normalization_list = [
      NormalizerSpace("vector"),
      NormalizerSpace("auc"),
      NormalizerSpace("vector"),
  ]


  classifier_list = [
      ClassifierSpace(SVC(),{"C": [0.1, 1, 10, 100], "gamma": [0.1, 0.01, 0.001, "scale"], "kernel": ["linear", "rbf"]})
  ]

  param_list = {
      DENOISING_STR: denoiser_list,
      BASELINE_STR: baseline_list,
      NORMALIZE_STR: normalization_list,
      CLASSIFIER_STR: classifier_list
  }

  return param_list


def extract_best_split_scores(cv_res, best_idx):
    # find all split test score columns
    split_cols = [c for c in cv_res if c.startswith("split") and c.endswith("_test_accuracy")]
    # sort them numerically
    split_cols = sorted(split_cols, key=lambda x: int(x.split('_')[0][5:]))

    # extract the per-split scores for the best hyperparameters
    return np.array([cv_res[col][best_idx] for col in split_cols])


if __name__ == "__main__":

    X, y, patients = ... # Setup the dataset, coid, melanoma, or covid

    # Crop the spectra
    # Ovarian region: (500,1800)
    # Melanoma region: (600, 1800)
    # Covid region: (600, 1800)
    pipeline = rpr.Pipeline([
        rpr.misc.Cropper(region=(500, 1800)),
    ])

    # Perform cropping to the dataset
    X_preprocessed = clean_dataframe(X, pipeline, return_dataframe=True)

    param_grid = setup_param_grid()

    RANDOM_SEEDS = [17, 42, 73, 101, 256, 389, 512, 777, 1024, 2025]

    accuracies = []

    for rs in RANDOM_SEEDS:
        print(f"Random Seed: {rs}")
        estimator = RamanPipeline()

        cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=rs)

        search = RamanSearch(estimator=estimator,
                             research_strategy=GridSearchStrategy(),
                             param_grid=param_grid,
                             cv=cv,
                             return_train_score=True,
                             compute_is_score=True,
                             n_jobs=1,
                             verbose=10,
                             refit="accuracy")

        start = time.perf_counter()
        res = search.fit(X_preprocessed, y, groups=patients)
        end = time.perf_counter()
        print(f"Fitted GridSearch for Ovarian dataset with random seed {rs}")

        accuracies.extend(extract_best_split_scores(res.cv_results_, res.best_index_))

    accuracies = np.array(accuracies)
    results = pd.DataFrame(accuracies, columns=['RamanPipeline'])
    results.to_csv(..., index=False)

