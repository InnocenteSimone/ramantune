import pandas as pd
from ramantune.search.search_space import DenoiserSpace, BaselineSpace, NormalizerSpace, ClassifierSpace, FeatureSelectionSpace
from ramantune.pipeline import RamanPipeline
from ramantune.search import RamanSearch
from ramantune.search.strategies import GridSearchStrategy

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA

from ramantune.utils.config import DENOISING_STR, BASELINE_STR, NORMALIZE_STR, FEATURE_SELECTION_STR, CLASSIFIER_STR

def setup_param_grid():
    denoiser_list = [
        DenoiserSpace("savgol", {"window_length": [7], "polyorder": [3]}),
    ]

    baseline_list = [
        BaselineSpace("imodpoly", {"poly_order": [3]}),
        BaselineSpace("bubblefill", {"min_bubble_widths": [100]})
    ]

    normalization_list = [
        NormalizerSpace("auc"),
        NormalizerSpace("snv")
    ]

    feature_selection_list = [
        FeatureSelectionSpace(PCA(), {"n_components": [0.99]}),
        FeatureSelectionSpace(None, None),
    ]

    classifier_list = [
        ClassifierSpace(SVC(), {"C": [1], "kernel": ["rbf"], "gamma": [0.01]}),
    ]

    param_list = {
        DENOISING_STR: denoiser_list,
        BASELINE_STR: baseline_list,
        NORMALIZE_STR: normalization_list,
        FEATURE_SELECTION_STR: feature_selection_list,
        CLASSIFIER_STR: classifier_list
    }

    return param_list

from ramantune.utils import register_algorithm
from ramantune.custom import RamanPipelineStep
from orpl.baseline_removal import bubblefill

@register_algorithm(category=NORMALIZE_STR, name="snv")
class SNVNormalization(RamanPipelineStep):
    def __init__(self):
        super().__init__(self.snv_normalization)

    @staticmethod
    def snv_normalization(spectral_data, spectral_axis):
        mean, std = spectral_data.mean(), spectral_data.std()
        return (spectral_data - mean) / std, spectral_axis

@register_algorithm(category=BASELINE_STR, name="bubblefill")
class BubbleFill(RamanPipelineStep):
    def __init__(self, *, min_bubble_widths=50, fit_order=1):
        super().__init__(
            self._bubblefill_call,
            min_bubble_widths=min_bubble_widths,
            fit_order=fit_order
        )

    @staticmethod
    def _bubblefill_call(spectral_data, spectral_axis, *args, **kwargs):
        raman, bubblefill_b = bubblefill(
            spectral_data,
            min_bubble_widths=kwargs.get("min_bubble_widths", 50),
            fit_order=kwargs.get("fit_order", 1))
        return raman, spectral_axis



if __name__ == "__main__":
    # Read Dataset
    df = pd.read_csv('bin/ovarian.csv')

    df = df[df['patient'].isin([1, 2, 3, 29, 30, 31])]
    groups = df['patient']

    y = df['label'].values
    X = df.drop(columns=['label', 'patient'])

    # Setup RamanPipeline estimator
    estimator = RamanPipeline()

    # Setup parameter grid for Raman Search
    param_grid = setup_param_grid()

    search = RamanSearch(estimator=estimator,
                         research_strategy=GridSearchStrategy(),
                         param_grid=param_grid,
                         cv=StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True),
                         return_train_score=True,
                         n_jobs=5,
                         verbose=10,
                         refit="accuracy",
                         compute_is_score=True)

    res = search.fit(X, y, groups=groups)
    y_pred = search.predict(X)

    print(search.get_best_params())
    print(search.get_best_score())

    search.get_cv_results(file_path=f"ovarian_add_preprocessing.csv",
                          return_split_scores=False,
                          return_combined_params=True,
                          round_values=False)




