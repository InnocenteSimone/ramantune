Search Space
============

Introduction
------------
The ``ramantune.search.search_space`` module provides composable building
blocks to construct hyperparameter search spaces for Raman analysis
pipelines. It defines a small set of ``SearchSpaceBlock`` subclasses that
represent configurable pipeline categories (for example denoising,
baseline correction, normalization, feature selection and classification).

Each block can emit either an unprefixed parameter mapping or a category-
prefixed mapping (``"{category}__{parameter}"``) that is directly
consumable by scikit-learn-style pipeline search utilities. These blocks
are intended to be composed into a ``param_grid`` which can then be used
by higher-level search orchestrators such as ``RamanSearch``.

Usage example
-------------
A minimal example showing how to assemble a parameter grid.

.. code-block:: python

    from ramantune.search.search_space.search_space import (
        DenoiserSpace, BaselineSpace, NormalizerSpace,
        FeatureSelectionSpace, ClassifierSpace
    )

    from ramantune.utils.config import (
        DENOISING_STR, BASELINE_STR,
        NORMALIZE_STR, FEATURE_SELECTION_STR, CLASSIFIER_STR
    )

    denoiser_list = [
        DenoiserSpace("savgol", {"window_length": [7], "polyorder": [3]}),
    ]

    baseline_list = [
        BaselineSpace("imodpoly", {"poly_order": [10]}),
        BaselineSpace("asls", {"lam": [1e3, 1e5]})
    ]

    normalization_list = [NormalizerSpace("auc")]

    feature_selection_list = [
        FeatureSelectionSpace(PCA(), {"n_components": [0.99, 12]}),
        FeatureSelectionSpace(None, None),
    ]

    classifier_list = [
        ClassifierSpace(SVC(), {"C": [1, 10], "kernel": ["rbf"], "gamma": [0.01]}),
    ]

    param_grid = {
        DENOISING_STR: denoiser_list,
        BASELINE_STR: baseline_list,
        NORMALIZE_STR: normalization_list,
        FEATURE_SELECTION_STR: feature_selection_list,
        CLASSIFIER_STR: classifier_list
    }

This ``param_grid`` can be passed to ``RamanSearch`` or to a strategy
setup method to generate the concrete candidate parameter dictionaries.

API Reference
-------------

.. automodule:: ramantune.search.search_space.search_space
   :members:
