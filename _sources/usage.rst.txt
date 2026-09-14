Usage
==============

Installation
------------
.. code-block:: bash

    pip install ramantune

Quick Start
-----------

1. **Import the necessary modules**

.. code-block:: python

    import pandas as pd
    from ramantune.pipeline import RamanPipeline
    from ramantune.search import RamanSearch
    from ramantune.search.strategies import GridSearchStrategy
    from ramantune.search.search_space import (
        DenoiserSpace, BaselineSpace, NormalizerSpace, 
        FeatureSelectionSpace, ClassifierSpace
    )
    from ramantune.utils.config import (
        DENOISING_STR, BASELINE_STR, NORMALIZE_STR, 
        FEATURE_SELECTION_STR, CLASSIFIER_STR
    )

    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold

2. **Prepare your data**

Ensure you have your dataset ready with features and labels properly separated.

.. code-block:: python

    df = pd.read_csv('your_data.csv')
    y = df['label'].values
    X = df.drop(columns=['label']).values

3. **Define the search space**

Define the search space for each block of the Raman Pipeline. Parameters are
automatically validated—if you specify invalid parameters, a descriptive error
will be raised.

.. code-block:: python

    denoiser_list = [
        DenoiserSpace("savgol", {"window_length": [7], "polyorder": [3]}),
    ]

    baseline_list = [
        BaselineSpace("imodpoly", {"poly_order": [10]}),
        BaselineSpace("asls", {"lam": [1e3, 1e5]})
    ]

    normalization_list = [
        NormalizerSpace("auc")
    ]

    feature_selection_list = [
        FeatureSelectionSpace(PCA(), {"n_components": [0.99, 12]}),
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

**Note:** All parameters are validated at initialization. If you specify an
invalid parameter, you'll get a helpful error message listing the valid options
for that algorithm.

4. **Setup pipeline and search**

Create a RamanPipeline and RamanSearch instance to find the best combination
of preprocessing steps and model hyperparameters.

.. code-block:: python

    estimator = RamanPipeline()

    search = RamanSearch(
        estimator=estimator,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        return_train_score=True,
        n_jobs=3,
        verbose=3,
        refit="accuracy"
    )

    search.fit(X, y)

5. **Retrieve results**

.. code-block:: python

    # Get best parameters and score
    best_params = search.get_best_params()
    best_score = search.get_best_score()

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    # Save all results to CSV
    search.get_cv_results(file_path="results.csv")
    results_df = pd.read_csv("results.csv")

Parameter Validation
--------------------

All search space blocks validate their parameters at initialization. This ensures
you catch configuration errors early with clear, actionable error messages.

.. code-block:: python

    from ramantune.search.search_space import DenoiserSpace, ClassifierSpace
    from sklearn.svm import SVC

    # Valid parameter - works silently
    space = DenoiserSpace("savgol", {"window_length": 7})

    # Invalid parameter - raises informative ValueError
    try:
        space = DenoiserSpace("savgol", {"invalid_param": 7})
    except ValueError as e:
        print(e)
        # Output: Invalid parameters for 'savgol': {'invalid_param'}. 
        #         Valid parameters: {'window_length', 'polyorder', 'mode', ...}

The validation works for all block types:

- **Preprocessing blocks** (Denoiser, Baseline, Normalizer): Validate against the
  ramanspy class ``__init__`` parameters
- **Sklearn blocks** (FeatureSelection, Classifier): Validate against the 
  estimator's ``get_params()``

Adding Custom Preprocessing Algorithms
---------------------------------------

You can extend Ramantune with custom preprocessing algorithms by implementing
a class and registering it.

1. **Create the custom class**

.. code-block:: python

    from ramantune.custom import RamanPipelineStep
    from ramantune.utils import register_algorithm
    from ramantune.utils.config import NORMALIZE_STR

    @register_algorithm(category=NORMALIZE_STR, name="snv")
    class SNVNormalization(RamanPipelineStep):
        """Standard Normal Variate normalization."""
        
        def __init__(self, scale=1.0):
            """
            Parameters
            ----------
            scale : float, optional
                Scaling factor (default: 1.0)
            """
            self.scale = scale
            super().__init__(self.snv_normalization)

        def snv_normalization(self, spectral_data, spectral_axis):
            mean, std = spectral_data.mean(), spectral_data.std()
            return (spectral_data - mean) / (std * self.scale), spectral_axis

2. **Use the custom algorithm in your search space**

.. code-block:: python

    normalization_list = [
        NormalizerSpace("auc"),
        NormalizerSpace("snv", {"scale": [1.0, 2.0]})
    ]

The custom algorithm will now be part of your parameter grid and will be
automatically validated along with other parameters.

Common Errors
-------------

**ValueError: Invalid parameters for 'SVC'**

This error occurs when you specify a parameter that doesn't exist for the algorithm.

.. code-block:: python

    # WRONG - 'invalid_param' is not a valid SVC parameter
    space = ClassifierSpace(SVC(), {"invalid_param": 1})

    # RIGHT - use valid SVC parameters
    space = ClassifierSpace(SVC(), {"C": 1, "kernel": "rbf"})

To see available parameters, check the algorithm's documentation or catch the
error message which lists all valid parameters.

**ValueError: Algorithm not found in category**

This error occurs when you specify an algorithm name that doesn't exist in the registry.

.. code-block:: python

    # WRONG - 'nonexistent' is not a registered denoising algorithm
    space = DenoiserSpace("nonexistent", {})

    # RIGHT - use a registered algorithm
    space = DenoiserSpace("savgol", {})

Check the registry or the API reference for available algorithms in each category.
