"""Abstract interface for hyperparameter research strategies.

This module defines the base contract implemented by concrete search
strategies. Implementations are expected to configure search spaces, execute
fitting, and expose cross-validation results.
"""
from abc import ABC, abstractmethod


class ResearchStrategy(ABC):
    """Define the strategy contract for model and preprocessing research.

    Attributes
    ----------
    research : object or None
        Backend search object (e.g., grid/randomized search instance)
        managed by concrete strategies.
    """

    def __init__(self):
        """Initialize strategy state."""
        self.research = None

    def get_research(self):
        """Return the strategy instance.

        Returns
        -------
        ResearchStrategy
            Current strategy object.
        """
        return self

    @abstractmethod
    def get_classifier_params(self, params: list[dict]):
        """Extract classifier-related search parameters.

        Parameters
        ----------
        params : list of dict
            Full parameter mapping provided by caller.

        Returns
        -------
        list of dict
            Classifier-specific parameter mapping.
        """
        pass

    @abstractmethod
    def get_preprocessing_params(self, params: list[dict]):
        """Extract preprocessing-related search parameters.

        Parameters
        ----------
        params : list of dict
            Full parameter mapping provided by caller.

        Returns
        -------
        list of dict
            Preprocessing-specific parameter mapping.
        """
        pass

    @abstractmethod
    def setup_search_space(self, params):
        """Build and configure the search space from input parameters.

        Parameters
        ----------
        params : dict
            User-defined settings used to construct search candidates.
        """
        pass

    @abstractmethod
    def fit_strategy(self, X, y=None, groups=None, **fit_params):
        """Fit the configured search backend.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like, optional
            Target labels.
        groups : array-like, optional
            Group labels for grouped cross-validation splitters.
        **fit_params : dict
            Additional keyword arguments passed to backend ``fit``.

        Returns
        -------
        object
            Fitted search backend or strategy-specific fit result.
        """
        pass

    @abstractmethod
    def get_cv_results(self):
        """Return cross-validation results for the executed search.

        Returns
        -------
        dict or pandas.DataFrame
            Cross-validation metrics and metadata, implementation-dependent.
        """
        pass
