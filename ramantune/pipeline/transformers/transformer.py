import pandas as pd
import ramanspy as rp
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

class SpectraTransformer(BaseEstimator, TransformerMixin):
    """Convert tabular spectra into ``ramanspy`` ``Spectrum`` objects.

    Notes
    -----
    Input columns are interpreted as the spectral axis and each row as one
    spectrum intensity vector.
    """

    def fit(self, X, y=None):
        """Return self without fitting state.

        Parameters
        ----------
        X : pandas.DataFrame
            Input spectral matrix.
        y : array-like, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        SpectraTransformer
            Unmodified estimator instance.
        """
        return self

    def transform(self, X, y=None):
        """Transform a DataFrame into a list of ``Spectrum`` objects.

        Parameters
        ----------
        X : pandas.DataFrame
            Input spectra where columns are Raman shifts and rows are samples.
        y : array-like, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        list of rp.Spectrum
            Converted spectra objects.
        """
        spectral_axis = X.columns.values.astype("float")
        spectral_data = X.values.astype("float")

        return [rp.Spectrum(data, spectral_axis) for data in spectral_data]

class InverseSpectraTransformer(BaseEstimator, TransformerMixin):
    """Convert ``Spectrum`` objects back to a tabular DataFrame."""

    def fit(self, X, y=None):
        """Return self without fitting state.

        Parameters
        ----------
        X : list
            Input spectra collection.
        y : array-like, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        InverseSpectraTransformer
            Unmodified estimator instance.
        """
        return self

    def transform(self, X, y=None):
        """Transform spectra collection into a DataFrame.

        Parameters
        ----------
        X : list of rp.Spectrum
            Spectra to be converted.
        y : array-like, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per spectrum and columns as spectral axis.
        """
        if not X:
            return pd.DataFrame()

        data_matrix = [s.spectral_data for s in X]
        axis = X[0].spectral_axis

        return pd.DataFrame(data_matrix, columns=axis)

class RamanPreprocessingTransformer(BaseEstimator, TransformerMixin):
    """Apply a selectable Raman preprocessing algorithm from a registry.

        Parameters
        ----------
        category : str
            Logical preprocessing category (e.g., denoising, baseline).
        algorithm : str, optional
            Registry key identifying the algorithm class to instantiate.
        registry : dict-like, optional
            Mapping from algorithm names to callable classes.
    """

    def __init__(self, category, algorithm=None, registry=None):
        """Initialize preprocessing transformer configuration."""
        self.category = category
        self.algorithm = algorithm
        self.alg_instance = None
        self.algo_params = {}
        self.registry = registry

    def _instantiate_algorithm(self):
        """Instantiate the configured algorithm with stored parameters.

        Returns
        -------
        object
            Instantiated preprocessing algorithm.

        Raises
        ------
        ValueError
            If no algorithm name has been configured.
        """
        if self.algorithm is None:
            raise ValueError("Algorithm must be set before fitting the model.")

        algo_cls = self.registry[self.algorithm]
        return algo_cls(**self.algo_params)

    def fit(self, X, y=None):
        """Create algorithm instance for subsequent transformations.

        Parameters
        ----------
        X : list of rp.Spectrum
            Input spectra.
        y : array-like, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        RamanPreprocessingTransformer
            Fitted transformer with initialized algorithm instance.
        """
        self.alg_instance = self._instantiate_algorithm()
        return self

    def transform(self, X, y=None):
        """Apply the configured preprocessing algorithm to each spectrum.

        Parameters
        ----------
        X : list of rp.Spectrum
            Input spectra.
        y : array-like, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        list of rp.Spectrum
            Transformed spectra. If ``algorithm`` is ``None``, returns ``X``.
        """
        if self.algorithm is None:
            return X
        return [self.alg_instance.apply(sp) for sp in X]

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform with the wrapped selector.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like, optional
            Target labels.
        **fit_params : dict
            Optional fit-time keyword arguments.

        Returns
        -------
        array-like
            Transformed features, or ``X`` when no selector is set.
        """
        if self.algorithm is None:
            return X
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **params):
        """Set transformer and algorithm parameters.

        Parameters
        ----------
        **params : dict
            Parameter mapping. ``algorithm`` updates the algorithm key;
            remaining entries are passed to the algorithm constructor.

        Returns
        -------
        RamanPreprocessingTransformer
            Transformer with updated parameters.
        """
        self.algorithm = params.pop("algorithm")
        if not self.algorithm is None:
            self.algo_params.update(params)
            self.alg_instance = self._instantiate_algorithm()

        return self

    def get_params(self, deep=True):
        """Get estimator parameters for scikit-learn compatibility.

        Parameters
        ----------
        deep : bool, optional
            Ignored; included for API compatibility.

        Returns
        -------
        dict
            Current transformer and algorithm parameters.
        """
        params = {"category": self.category, "algorithm": self.algorithm, "registry": self.registry}
        params.update(self.algo_params)
        return params

class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """Wrap a feature-selection estimator for pipeline integration.

    Parameters
    ----------
    algorithm : object, optional
        Feature selection estimator implementing ``fit`` and ``transform``.
    """
    def __init__(self, algorithm=None):
        self.algorithm = algorithm

    def fit(self, X, y=None):
        """Fit the wrapped feature-selection algorithm.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like, optional
            Target labels.

        Returns
        -------
        FeatureSelectionTransformer
            Fitted transformer.

        Raises
        ------
        ValueError
            If ``algorithm`` is not set.
        """
        if self.algorithm is None:
            raise ValueError("Algorithm must be set before fitting the model.")
        self.algorithm.fit(X, y)
        return self

    def transform(self, X, y=None):
        """Transform features using wrapped selector when available.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like, optional
            Ignored.

        Returns
        -------
        array-like
            Selected/transformed features, or ``X`` when no selector is set.
        """
        if self.algorithm is not None:
            return self.algorithm.transform(X)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform with the wrapped selector.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like, optional
            Target labels.
        **fit_params : dict
            Optional fit-time keyword arguments.

        Returns
        -------
        array-like
            Transformed features, or ``X`` when no selector is set.
        """
        if self.algorithm is None:
            return X

        if hasattr(self.algorithm, "fit_transform"):
            return self.algorithm.fit_transform(X, y)
        self.algorithm.fit(X, y)
        return self.algorithm.transform(X)

    def get_params(self, deep=True):
        """Get estimator parameters.

        Parameters
        ----------
        deep : bool, optional
            Ignored; included for API compatibility.

        Returns
        -------
        dict
            Parameter dictionary containing ``algorithm``.
        """
        return {"algorithm": self.algorithm}

    def set_params(self, **params):
        """Set wrapped feature-selection estimator and its parameters.

        Parameters
        ----------
        **params : dict
            Parameter mapping where ``algorithm`` is the wrapped estimator.

        Returns
        -------
        FeatureSelectionTransformer
            Updated transformer.
        """
        alg = params.pop("algorithm")
        if not alg is None:
            alg.set_params(**params)
            setattr(self, "algorithm", alg)
        return self

class ModelTransformer(BaseEstimator, ClassifierMixin):
    """Wrap a classifier for final pipeline stage compatibility.

    Parameters
    ----------
    algorithm : object, optional
        Classifier implementing ``fit``/``predict`` and optional
        ``predict_proba``/``score``.
    """

    def __init__(self, algorithm=None):
        """Initialize model wrapper."""
        self.algorithm = algorithm

    def transform(self, X):
        """Pass features through unchanged.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        array-like
            Unchanged input ``X``.
        """
        return X

    def fit(self, X, y=None):
        """Fit the wrapped classifier.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like, optional
            Target labels.

        Returns
        -------
        ModelTransformer
            Fitted classifier wrapper.

        Raises
        ------
        ValueError
            If ``algorithm`` is not set.
        """
        if self.algorithm is None:
            raise ValueError("Algorithm must be set before fitting the model.")

        self.algorithm.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X, y=None):
        """Predict labels with the wrapped classifier.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like, optional
            Ignored.

        Returns
        -------
        array-like
            Predicted labels.
        """
        return self.algorithm.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities with the wrapped classifier.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        array-like
            Class probability estimates.
        """
        return self.algorithm.predict_proba(X)

    def score(self, X, y):
        """Return classifier score when available.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like
            Target labels.

        Returns
        -------
        float or None
            Score returned by wrapped classifier, or ``None`` if no algorithm
            is configured.
        """
        return self.algorithm.score(X, y) if self.algorithm is not None else None

    def get_params(self, deep=True):
        """Get estimator parameters.

        Parameters
        ----------
        deep : bool, optional
            Ignored; included for API compatibility.

        Returns
        -------
        dict
            Parameter dictionary containing ``algorithm``.
        """
        return {"algorithm": self.algorithm}

    def set_params(self, **params):
        """Set wrapped classifier and nested parameters.

        Parameters
        ----------
        **params : dict
            Parameter mapping where ``algorithm`` is the wrapped classifier.

        Returns
        -------
        ModelTransformer
            Updated wrapper.
        """
        alg = params.pop("algorithm")
        if alg is not None and hasattr(alg, "set_params"):
            alg.set_params(**params)
        setattr(self, "algorithm", alg)
        return self
