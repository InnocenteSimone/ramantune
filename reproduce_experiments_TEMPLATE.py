import os
import re
import time
import numpy as np
import pandas as pd
import ramanspy as rp
import ramanspy.preprocessing as rpr
from pathlib import Path

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
from ramantune.pipeline.raman_pipeline import RamanPipeline
from ramantune.search.search_space import DenoiserSpace, BaselineSpace, NormalizerSpace, ClassifierSpace, FeatureSelectionSpace
from ramantune.utils.config import DENOISING_STR, BASELINE_STR, NORMALIZE_STR, FEATURE_SELECTION_STR, CLASSIFIER_STR
from ramantune.search.strategies import GridSearchStrategy
from ramantune.search import RamanSearch

def ovarian_rows(folder, label, offset):
    rows = []

    for path in sorted(folder.glob("*.txt")):
        match = re.search(r"plasma_(HC|OC)_(\d+)_", path.name)
        source_patient_id = int(match.group(2))

        raw = np.loadtxt(path)[:1480]  # drops extra points in 11 files
        axis = raw[:, 0]
        intensity = raw[:, 1]

        # Raw files are high-to-low shift; make feature columns ascending.
        rows.append({
            **dict(zip(axis[::-1], intensity[::-1])),
            "patient": source_patient_id + offset,
            "label": label,
        })

    return rows


def format_ovarian_dataset():
    FILEPATH_OVARIAN = "..." # Path to the ovarian cancer dataset
    root = Path(FILEPATH_OVARIAN)

    out = Path("bin")
    out.mkdir(exist_ok=True)

    rows = (
            ovarian_rows(root / "Healthy", "HC", offset=0)
            + ovarian_rows(root / "Ovarian_Cancer", "OC", offset=28)
    )

    df = pd.DataFrame(rows)
    feature_columns = sorted(c for c in df.columns if isinstance(c, float))
    df = df[feature_columns + ["patient", "label"]]

    df.to_csv(out / "ovarian.csv", index=False)

    # The compact, 42-spectrum dataset used by example.ipynb:
    small = df[df["patient"].isin([1, 2, 3, 29, 30, 31])]
    small.to_csv(out / "ovarian_small.csv", index=False)

    print(df.shape)  # (385, 1482): 1480 spectral columns + patient + label
    print(small.shape)  # (42, 1482)

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
      BaselineSpace("asls", {"lam": [100]}),
  ]

  normalization_list = [
      NormalizerSpace("snv"),
      NormalizerSpace("vector"),
  ]


  classifier_list = [
      ClassifierSpace(SVC(),{"C": [1, 10, 100], "gamma": [0.1, 0.001, "scale"], "kernel": ["linear", "rbf"]})
  ]

  param_list = {
      DENOISING_STR: denoiser_list,
      BASELINE_STR: baseline_list,
      NORMALIZE_STR: normalization_list,
      CLASSIFIER_STR: classifier_list
  }

  return param_list


if __name__ == "__main__":

    dataset = "ovarian" # covid,melanoma

    df = format_ovarian_dataset() # Setup the dataset, coid, melanoma, or covid
    patients = df['patient'].values
    y = df['label'].values
    X = df.drop(columns=['label', 'patient']).valeus

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

    for rs in RANDOM_SEEDS:
        print(f"Random Seed: {rs}")
        FILEPATH_STORE_RESULTS = f'{dataset}_nested_{rs}.csv'

        outer_results = []
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)

        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_preprocessed, y), start=1):
            print(f"\n{'=' * 80}")
            print(f"Outer fold {outer_fold}/10")
            print(f"{'=' * 80}")
            X_train = X_preprocessed.iloc[train_idx]
            X_test = X_preprocessed.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            groups_train = patients[train_idx]
            groups_test = patients[test_idx]

            inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rs + 1000)

            estimator = RamanPipeline()

            search = RamanSearch(
                estimator=estimator,
                research_strategy=GridSearchStrategy(),
                param_grid=param_grid,
                cv=inner_cv,
                return_train_score=True,
                n_jobs=2,
                verbose=10,
                refit="accuracy",
                compute_is_score=True
            )

            start = time.perf_counter()
            search.fit(X_train, y_train, groups=groups_train)
            end = time.perf_counter()

            print(f"Inner HPO completed in {(end - start) / 60:.2f} min")

            best_model = search.research.get_research().best_estimator_

            y_pred = best_model.predict(X_test)

            outer_accuracy = np.mean(y_pred == y_test)

            print(f"Outer fold {outer_fold} accuracy: {outer_accuracy:.4f}")

            outer_results.append({
                "seed": rs,
                "outer_fold": outer_fold,
                "accuracy": outer_accuracy,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "best_params": search.research.get_research().best_params_
            })

            cv_res = pd.DataFrame(search.get_cv_results())
            cv_res.to_csv(f"nested_cv_results/{dataset}/outer/cv_res_nested_outer_fold_{outer_fold}_{dataset}_{rs}.csv",
                          index=False)

        results = pd.DataFrame(outer_results)

        results.to_csv(FILEPATH_STORE_RESULTS, index=False)

        print("\nNested CV results:")
        print(results[["outer_fold", "accuracy", "n_train", "n_test"]])
        print(f"Std outer accuracy: {results['accuracy'].std():.4f}")
        print("Ended")
