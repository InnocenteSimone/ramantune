Raman Search
============

Introduction
------------
The ``ramantune.search.RamanSearch`` module provides a high-level orchestration
layer for hyperparameter search and model evaluation of Raman analysis
pipelines. It coordinates three main pieces:

- A pipeline wrapper (``RamanPipeline``) that builds preprocessing
  and classifier steps.
- A search-space description composed of ``SearchSpaceBlock`` objects (see
  :mod:`ramantune.search.search_space.search_space`) that define candidate
  algorithms and parameter grids per pipeline category.
- A research strategy backend encapsulated by the strategy API (``GridSearchStrategy`` by
  default).

The class exposes convenience methods to build candidate parameter lists,
configure scoring (including a custom ``is_score`` metric), run cross-
validation search, and extract CV results or predictions from the best
found estimator.

Usage example
-------------
A minimal example demonstrating how to construct a parameter grid, create a
``RamanPipeline`` and launch a search.

.. code-block:: python

    from ramantune.pipeline import RamanPipeline
    from ramantune.search import RamanSearch
    from ramantune.search.search_space.search_space import (
        DenoiserSpace, BaselineSpace, NormalizerSpace,
        FeatureSelectionSpace, ClassifierSpace
    )
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedGroupKFold

    # Assume X_train, y_train, groups are defined

    # Assemble a param_grid using the provided SearchSpace blocks

    denoiser_list = [
        DenoiserSpace("savgol", {"window_length": [7], "polyorder": [3]}),
        DenoiserSpace(None) # No Denoising
    ]

    baseline_list = [
        BaselineSpace("imodpoly", {"poly_order": [4]}),
        BaselineSpace("asls", {"lam": [100]}),
    ]

    normalization_list = [
        NormalizerSpace("vector"),
    ]

    feature_selection_list = [
        FeatureSelectionSpace(None), # No feature selection
        FeatureSelectionSpace(PCA(),{"n_components": [0.90, 10]}),
    ]

    classifier_list = [
        ClassifierSpace(SVC(),{"C": [0.1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale"]})
    ]

    param_grid = {
        DENOISING_STR: denoiser_list,
        BASELINE_STR: baseline_list,
        NORMALIZE_STR: normalization_list,
        FEATURE_SELECTION_STR: feature_selection_list,
        CLASSIFIER_STR: classifier_list
    }

    # Create pipeline and search
    pipeline = RamanPipeline()
    search = RamanSearch(estimator=pipeline,
                         param_grid=param_grid,
                         cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0),
                         compute_is_score=True)

    # Fit the search and retrieve results
    search.fit(X_train, y_train, groups=groups)
    results = search.get_cv_results()  # returns a dict or can write to CSV

API Reference
-------------

.. automodule:: ramantune.search.raman_search
   :members:
