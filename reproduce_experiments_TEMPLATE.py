import time
import numpy as np
import pandas as pd
import ramanspy as rp
import ramanspy.preprocessing as rpr
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC

from ramantune.pipeline.raman_pipeline import RamanPipeline
from ramantune.search.search_space import DenoiserSpace, BaselineSpace, NormalizerSpace, ClassifierSpace, FeatureSelectionSpace
from ramantune.utils.config import DENOISING_STR, BASELINE_STR, NORMALIZE_STR, FEATURE_SELECTION_STR, CLASSIFIER_STR
from ramantune.search.strategies import GridSearchStrategy
from ramantune.search import RamanSearch

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

