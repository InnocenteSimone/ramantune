import time
import pandas as pd
from ramantune.pipeline.raman_pipeline import RamanPipeline

from ramantune.search.search_space import DenoiserSpace, BaselineSpace, NormalizerSpace, ClassifierSpace, FeatureSelectionSpace
from ramantune.utils.config import DENOISING_STR, BASELINE_STR, NORMALIZE_STR, FEATURE_SELECTION_STR, CLASSIFIER_STR
from ramantune.search.strategies import GridSearchStrategy
from ramantune.search import RamanSearch

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold



def setup_param_grid():
    denoiser_list = [
        DenoiserSpace("savgol", {"window_length": [7, 11], "polyorder": [3]}),
    ]

    baseline_list = [
        BaselineSpace("imodpoly", {"poly_order": [4]}),
        BaselineSpace("asls", {"lam": [100]}),
    ]

    normalization_list = [
        NormalizerSpace("vector"),
        NormalizerSpace(None) # No Normalization
    ]

    feature_selection_list = [
        FeatureSelectionSpace(None, None),         # No feature selection
        FeatureSelectionSpace(PCA(),{"n_components": [0.90, 10]}),
    ]

    classifier_list = [
        ClassifierSpace(SVC(),{"C": [0.1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale"]})
    ]

    param_list = {
        DENOISING_STR: denoiser_list,
        BASELINE_STR: baseline_list,
        NORMALIZE_STR: normalization_list,
        FEATURE_SELECTION_STR: feature_selection_list,
        CLASSIFIER_STR: classifier_list
    }

    return param_list

if __name__ == "__main__":

    # Read Dataset
    df = pd.read_csv('bin/ovarian.csv')

    df = df[df['patient'].isin([1, 2, 3, 29, 30, 31])]
    groups = df['patient']

    y = df['label'].values
    X = df.drop(columns=['label','patient'])

    estimator = RamanPipeline()
    param_grid = setup_param_grid()

    search = RamanSearch(estimator=estimator,
                         research_strategy=GridSearchStrategy(),
                         param_grid=param_grid,
                         cv=StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True),
                         return_train_score=True,
                         n_jobs=5,
                         verbose=10,
                         refit="accuracy")

    res = search.fit(X, y, groups=groups)

    y_pred = search.predict(X)
    print(res)

    print(search.get_best_params())
    print(search.get_best_score())

    search.get_cv_results(file_path=f"supplemeantary_info_ovarian.csv",
                          return_split_scores=False,
                          return_combined_params=True,
                          round_values=False)