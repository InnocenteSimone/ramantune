from .context.abstract import ResearchStrategy
from sklearn.model_selection import GridSearchCV
from ramantune.utils import ALGORITHM_STR, CLASSIFIER_STR

class GridSearchStrategy(ResearchStrategy, GridSearchCV):
    """Concrete strategy wrapping scikit-learn grid search.

        Notes
        -----
        The class inherits from both ``ResearchStrategy`` and ``GridSearchCV`` to
        expose a unified strategy interface while preserving native grid-search
        behavior.
    """

    def __init__(self):
        GridSearchCV.__init__(self, estimator=None, param_grid={})
        ResearchStrategy.__init__(self)

    def get_classifier_params(self, params):
        """Updated grids where classifier hyperparameters with a single value
           are pre-applied to the classifier instance, then removed from the
           explicit grid.

        Parameters
        ----------
        params : list of dict
            Candidate parameter grids.

        Returns
        -------
        list of dict
        """
        for d in params:
            to_erase = []
            for hyperparameter, values in d.items():
                if hyperparameter == f"{CLASSIFIER_STR}__{ALGORITHM_STR}":
                    continue

                if len(values) == 1:
                    d[f"{CLASSIFIER_STR}__{ALGORITHM_STR}"] = d[f"{CLASSIFIER_STR}__{ALGORITHM_STR}"].set_params(
                        **{hyperparameter.replace(f"{CLASSIFIER_STR}__", ""): values[0]}
                    )
                    to_erase.append(hyperparameter)

            for el in to_erase:
                d.pop(el)

        return params

    def get_preprocessing_params(self, params):
        """Return preprocessing parameter grids unchanged.

        Parameters
        ----------
        params : list of dict
            Preprocessing parameter grids.

        Returns
        -------
        list of dict
            Unmodified input grids.
        """
        return params

    def get_research(self):
        """Return underlying search object.

        Returns
        -------
        GridSearchStrategy
            Current strategy instance.
        """
        return self

    def get_cv_results(self):
        """Return cross-validation results produced by grid search.

        Returns
        -------
        dict
            ``GridSearchCV.cv_results_`` mapping.
        """
        return self.cv_results_

    def setup_search_space(self, search_space):
        """Assign parameter grid used by exhaustive search.

        Parameters
        ----------
        search_space : dict or list of dict
            Grid-search parameter space.
        """
        self.param_grid = search_space

    def fit_strategy(self, X, y=None, groups=None, **fit_params):
        """Fit grid search using strategy interface.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like, optional
            Target labels.
        groups : array-like, optional
            Group labels for grouped CV splitters.
        **fit_params : dict
            Extra keyword arguments forwarded to ``fit``.

        Returns
        -------
        GridSearchStrategy
            Fitted strategy instance.
        """
        return self.fit(X, y, groups=groups, **fit_params)

    def fit(self, X, y=None, groups=None, **fit_params):
        """Fit underlying grid-search estimator.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like, optional
            Target labels.
        groups : array-like, optional
            Group labels for grouped CV splitters.
        **fit_params : dict
            Extra keyword arguments forwarded to parent ``fit``.

        Returns
        -------
        GridSearchStrategy
            Fitted strategy instance.
        """
        super(GridSearchCV, self).fit(X, y, groups=groups, **fit_params)
        return self
