from .abstract import ResearchStrategy

class ResearchContext:
    """Delegate hyperparameter research operations to a strategy instance.

    Parameters
    ----------
    strategy : ResearchStrategy
        Initial concrete strategy used to perform search operations.

    Attributes
    ----------
    research_strategy : ResearchStrategy
        Active strategy object receiving delegated method calls.
    """

    def __init__(self, strategy: ResearchStrategy):
        """Initialize context with an initial research strategy.

        Parameters
        ----------
        strategy : ResearchStrategy
            Strategy implementation used by this context.
        """
        self.research_strategy = strategy

    def set_research_strategy(self, strategy: ResearchStrategy):
        """Replace the active research strategy.

        Parameters
        ----------
        strategy : ResearchStrategy
            New strategy implementation.
        """
        self.research_strategy = strategy

    def set_estimator(self, estimator):
        """Set the estimator on the underlying strategy.

        Parameters
        ----------
        estimator : object
            Estimator assigned to the strategy as ``estimator`` attribute.
        """
        self.research_strategy.__setattr__('estimator', estimator)

    def get_classifier_params(self, params=None):
        """Return classifier-related parameters from the active strategy.

        Parameters
        ----------
        params : dict, optional
            Full parameter mapping.

        Returns
        -------
        dict
            Classifier-specific parameter mapping.
        """
        return self.research_strategy.get_classifier_params(params)

    def get_preprocessing_params(self, params=None):
        """Return preprocessing-related parameters from the strategy.

        Parameters
        ----------
        params : dict, optional
            Full parameter mapping.

        Returns
        -------
        dict
            Preprocessing-specific parameter mapping.
        """
        return self.research_strategy.get_preprocessing_params(params)

    def setup_search_space(self, search_space):
        """Configure strategy search space.

        Parameters
        ----------
        search_space : dict or object
            Strategy-dependent search-space description.

        Returns
        -------
        object
            Strategy-specific setup result.
        """
        return self.research_strategy.setup_search_space(search_space)

    def get_research(self):
        """Return research backend from the active strategy.

        Returns
        -------
        object
            Strategy-managed search backend.
        """
        return self.research_strategy.get_research()

    def get_cv_results(self):
        """Return cross-validation results from the strategy.

        Returns
        -------
        dict or pandas.DataFrame
            Strategy-provided CV results.
        """
        return self.research_strategy.get_cv_results()

    def get_best_estimator(self):
        """Return the best estimator found by the strategy.

        Returns
        -------
        object
            Best estimator according to the executed search.
        """
        return self.research_strategy.best_estimator_

    def fit_research(self, X, y=None, *, groups=None, **fit_params):
        """Fit search process using the active strategy.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like, optional
            Target labels.
        groups : array-like, optional
            Group labels for grouped CV splitters.
        **fit_params : dict
            Additional keyword arguments passed to strategy fit.

        Returns
        -------
        object
            Strategy-specific fit result.
        """
        return self.research_strategy.fit_strategy(X, y, groups=groups, **fit_params)

    def get_best_index(self):
        """Return index of the best parameter set.

        Returns
        -------
        int
            Index of the best candidate in CV results.
        """
        return self.research_strategy.best_index_



