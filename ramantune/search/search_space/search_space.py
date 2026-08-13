import inspect
from ramantune.utils import default_registry
from sklearn.base import ClassifierMixin, RegressorMixin

from ramantune.utils import (
    ALGORITHM_STR,
    DENOISING_STR,
    BASELINE_STR,
    NORMALIZE_STR,
    FEATURE_SELECTION_STR,
    CLASSIFIER_STR)

class SearchSpaceBlock:
    """Represent one configurable category within a search space.

    This class provides a flexible interface for defining hyperparameter search
    spaces across different pipeline categories (denoising, baseline correction,
    normalization, feature selection, and classification). It automatically
    validates parameters to ensure they are valid for the selected algorithm.

    Parameters
    ----------
    category : str
        Pipeline category identifier (e.g., 'denoising', 'baseline', 'normalize',
        'feature_selection', 'classifier').
    algorithm_name : str, optional
        Registry key for the algorithm in the specified category. Used for
        registry-based algorithms (preprocessing steps). Default is None.
    params : dict, optional
        Hyperparameter mapping for the selected algorithm. All parameters are
        validated at initialization. Default is an empty dict.
    algorithm_function : object, optional
        Direct algorithm instance used when ``algorithm_name`` is ``None``.
        Typically used for sklearn estimators (feature selection, classifiers).
        Default is None.
    """

    def __init__(self, category, algorithm_name=None, params=None, algorithm_function=None):
        """Initialize a search-space block for one pipeline category."""
        self.category = category
        self.algorithm_name = algorithm_name
        self.params = params if params is not None else {}
        self.algorithm_function = default_registry.get_algorithm(category, algorithm_name) if algorithm_name is not None else algorithm_function

        if self.algorithm_function == -1:
            raise ValueError(
                f"Algorithm '{algorithm_name}' not found in category '{category}'. "
                f"Available: {default_registry.get_available_algorithms(category)}"
            )
        
        self._validate_params()

    def _validate_params(self):
        """Validate that all parameters are valid for the algorithm.
        
        Raises
        ------
        ValueError
            If any parameter is not valid for the algorithm.
        """
        if not self.params:
            return
        
        valid_params = self._get_valid_params()
        invalid_params = set(self.params.keys()) - valid_params
        
        if invalid_params:
            raise ValueError(
                f"Invalid parameters for '{self.algorithm_name or self.algorithm_function.__class__.__name__}': "
                f"{invalid_params}. Valid parameters: {valid_params}"
            )
    
    def _get_valid_params(self):
        """Get the set of valid parameter names for the algorithm.
        
        For sklearn estimators, uses get_params().
        For registry-based algorithms, uses inspect to get __init__ parameters.
        
        Returns
        -------
        set
            Valid parameter names for the algorithm.
        """
        # sklearn estimators have get_params method
        if hasattr(self.algorithm_function, 'get_params'):
            return set(self.algorithm_function.get_params(deep=False).keys())
        
        # Registry-based algorithms (ramanspy preprocessing classes)
        if inspect.isclass(self.algorithm_function):
            sig = inspect.signature(self.algorithm_function.__init__)
            return set(sig.parameters.keys()) - {'self'}
        
        # Fallback for other types
        return set()

    def get_param_grid_with_category(self):
        """Return category-prefixed parameter grid for pipeline usage.

        Returns
        -------
        dict
            Parameter mapping with keys prefixed as
            ``"{category}__{parameter}"``.

        Notes
        -----
        When ``algorithm_name`` is ``None``, ``algorithm_function`` is emitted
        directly as the algorithm value.
        """
        if self.algorithm_name is None:
            return {f"{self.category}__{ALGORITHM_STR}": self.algorithm_function,
                    **{f"{self.category}__{k}": v for k, v in self.params.items()}}
        return {f"{self.category}__{ALGORITHM_STR}": self.algorithm_name,
                **{f"{self.category}__{k}": v for k, v in self.params.items()}}

class DenoiserSpace(SearchSpaceBlock):
    """Search-space block for denoising methods.

    Represents a denoising preprocessing step with configurable algorithms
    from the ramanspy preprocessing registry.

    Parameters
    ----------
    algorithm : str
        Registered denoising algorithm name. Available options depend on the
        registry (e.g., 'savgol', 'whittaker', 'kernel', 'gaussian').
    params : dict, optional
        Algorithm hyperparameters. All parameters are validated against the
        algorithm's valid parameters. Default is None.
    """

    def __init__(self, algorithm, params=None):
        """Initialize denoising search-space configuration."""
        super().__init__(DENOISING_STR, algorithm, params)

class BaselineSpace(SearchSpaceBlock):
    """Search-space block for baseline correction methods.

    Represents a baseline correction preprocessing step with configurable
    algorithms from the ramanspy preprocessing registry.

    Parameters
    ----------
    algorithm : str
        Registered baseline algorithm name. Available options depend on the
        registry (e.g., 'asls', 'iasls', 'airpls', 'imodpoly', 'poly').
    params : dict, optional
        Algorithm hyperparameters. All parameters are validated against the
        algorithm's valid parameters. Default is None.
    """

    def __init__(self, algorithm, params=None):
        """Initialize baseline-correction search-space configuration."""
        super().__init__(BASELINE_STR, algorithm, params)

class NormalizerSpace(SearchSpaceBlock):
    """Search-space block for normalization methods.

    Represents a normalization preprocessing step with configurable algorithms
    from the ramanspy preprocessing registry.

    Parameters
    ----------
    algorithm : str
        Registered normalization algorithm name. Available options depend on the
        registry (e.g., 'vector', 'minmax', 'max_intensity', 'auc').
    params : dict, optional
        Algorithm hyperparameters. All parameters are validated against the
        algorithm's valid parameters. Default is None.
    """

    def __init__(self, algorithm, params=None):
        """Initialize normalization search-space configuration."""
        super().__init__(NORMALIZE_STR, algorithm, params)

class FeatureSelectionSpace(SearchSpaceBlock):
    """Search-space block for feature-selection estimators.

    Represents a feature selection step using sklearn-compatible estimators.
    Unlike preprocessing blocks, this takes a direct estimator instance rather
    than an algorithm name from the registry.

    Parameters
    ----------
    algorithm : object
        Feature-selection estimator instance. Must be a sklearn-compatible
        estimator with ``get_params()`` and ``set_params()`` methods.
        Common choices: ``PCA``, ``SelectKBest``, ``SelectPercentile``, etc.
    params : dict, optional
        Estimator hyperparameters. All parameters are validated using the
        estimator's ``get_params()`` method. Default is None.
    """

    def __init__(self, algorithm, params=None):
        """Initialize feature-selection search-space configuration."""
        super().__init__(FEATURE_SELECTION_STR, None, params, algorithm_function=algorithm)


class ClassifierSpace(SearchSpaceBlock):
    """Search-space block for classifier estimators.

    Represents a classification step using sklearn-compatible estimators.
    Accepts either ``ClassifierMixin`` or ``RegressorMixin`` instances,
    allowing flexibility in model selection.

    Parameters
    ----------
    algorithm : sklearn.base.ClassifierMixin or sklearn.base.RegressorMixin
        Classifier or regressor estimator instance. Must implement the
        sklearn estimator interface with ``fit()``, ``predict()``, ``get_params()``,
        and ``set_params()`` methods.
        Common choices: ``SVC``, ``RandomForestClassifier``, ``LogisticRegression``,
        ``SVR``, ``RandomForestRegressor``, etc.
    params : dict, optional
        Estimator hyperparameters. All parameters are validated using the
        estimator's ``get_params()`` method. Default is None.
    """

    def __init__(self, algorithm, params=None):
        """Initialize classifier search-space configuration."""
        if (not isinstance(algorithm, ClassifierMixin)) and (not isinstance(algorithm, RegressorMixin)):
            raise ValueError(f"Algorithm '{algorithm}' is not a valid classifier or regressor.")
        super().__init__(CLASSIFIER_STR, None, params, algorithm_function=algorithm)
