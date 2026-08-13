import itertools
import numpy as np
import pandas as pd
from functools import partial
from .strategies import GridSearchStrategy
from .strategies.context import ResearchContext

from ramantune.utils import CLASSIFIER_STR, ALGORITHM_STR
from ramantune.custom.custom_score import is_score, patient_accuracy
from ramantune.utils.decorators import measure_time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import specificity_score, sensitivity_score

class RamanSearch:
    """Coordinate estimator search space, scoring, and fitting workflow.

        Parameters
        ----------
        estimator : object, optional
            Raman pipeline wrapper exposing ``get_pipeline()``, ``get_steps()``,
            and ``build_pipeline(param_grid)``.
        research_strategy : object, optional
            Concrete strategy implementation used by ``ResearchContext``.
            Defaults to ``GridSearchStrategy()``.
        param_grid : dict, optional
            Category-indexed search-space blocks used to generate candidate
            parameter combinations.
        scoring : dict or callable, optional
            Scoring configuration compatible with scikit-learn search objects.
            If ``None``, defaults are created by :meth:`_setup_score`.
        n_jobs : int, optional
            Number of parallel workers for search backend.
        refit : bool or str, optional
            Refit strategy for best model selection.
        cv : object, optional
            Cross-validation splitter. Defaults to 10-fold stratified CV.
        verbose : int, optional
            Verbosity level for search backend.
        pre_dispatch : str or int, optional
            Controls job pre-dispatch behavior.
        error_score : float or "raise", optional
            Score assigned when fitting fails.
        return_train_score : bool, optional
            Whether to include training scores in CV results.
        compute_is_score : bool, optional
            Whether to include custom ``is_score`` in scoring dictionary.
    """
    def __init__(
            self,
            estimator=None,
            research_strategy=GridSearchStrategy(),
            param_grid=None,
            scoring=None,
            n_jobs=None,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch="2*n_jobs",
            error_score=np.nan,
            return_train_score=True,
            compute_is_score=False,
    ):
        self.research = ResearchContext(research_strategy)
        self.estimator = self.set_estimator(estimator)
        self.param_list = self.setup_research_params(param_grid)
        self._fitted = False

        self.set_additional_params(scoring=scoring,
                                   n_jobs=n_jobs,
                                   refit=refit,
                                   cv=cv,
                                   verbose=verbose,
                                   pre_dispatch=pre_dispatch,
                                   error_score=error_score,
                                   return_train_score=return_train_score,
                                   compute_is_score=compute_is_score)

    def set_estimator(self, estimator):
        # TODO: Check that estimator is an instance of RamanPipeline
        # TODO: Check that the transformers are the same of the params
        """
        Set estimator and propagate underlying pipeline to strategy.

        Parameters
        ----------
        estimator : object
            Estimator wrapper exposing ``get_pipeline()``.

        Returns
        -------
        object
            The same estimator instance for chaining.
        """
        self.research.set_estimator(estimator.get_pipeline())
        return estimator

    @staticmethod
    def update_dict(d):
        """
        Wrap algorithm entries in lists for search backend compatibility.

        Parameters
        ----------
        d : dict
            Candidate parameter dictionary.

        Returns
        -------
        dict
            Updated dictionary where algorithm values are list-wrapped.
        """
        keys = [k for k in d.keys() if ALGORITHM_STR in k]
        for key in keys:
            d[key] = [d[key]]
        return d

    def __prepare_params(self, param_grid):
        """
        Generate all candidate combinations across category spaces.

        Parameters
        ----------
        param_grid : dict
            Category-indexed list of parameter dictionaries.

        Returns
        -------
        list of dict
            Flattened list of merged parameter candidates.
        """
        combinations = itertools.product(*param_grid.values())
        candidate_list = []

        for algo_comb in combinations:
            candidate = {}
            for param in algo_comb:
                if param is not None:
                    candidate.update(param)
            candidate_list.append(self.update_dict(candidate))

        return candidate_list

    def setup_research_params(self, param_grid):
        """
        Build final candidate parameter list from category blocks.

        Parameters
        ----------
        param_grid : dict
            Mapping from pipeline category to iterable of search-space blocks.

        Returns
        -------
        list of dict
            Candidate parameter dictionaries consumable by search backend.

        Notes
        -----
        If current estimator steps differ from ``param_grid`` keys, the
        underlying pipeline is rebuilt to ensure consistency.
        """
        if self.estimator.get_steps() != list(param_grid.keys()):
            self.estimator.build_pipeline(param_grid)
            self.set_estimator(self.estimator)

        final_grid = {}

        for category, algorithms in param_grid.items():
            tmp = [alg.get_param_grid_with_category() for alg in algorithms]
            # We pass through research context to setup the params because for example Bayesian Optimization
            # does not support the same input format as GridSearchCV, so we need to convert it in the right format
            step_parameters = self.research.get_preprocessing_params(tmp) if category != CLASSIFIER_STR else self.research.get_classifier_params(tmp)
            final_grid[category] = step_parameters

        self.param_list = self.__prepare_params(final_grid)
        self.research.setup_search_space(search_space=self.param_list)

        return self.param_list

    def set_additional_params(self, **params):
        """
        Set strategy-backend parameters after initialization.

        Parameters
        ----------
        **params : dict
            Search backend options. Missing values are replaced with defaults.

        Notes
        -----
        Defaults include a multiclass scoring dictionary, stratified 10-fold
        CV, and training score reporting.
        """

        scoring = self._setup_score() if params.get("scoring", None) is None else params["scoring"]
        if params.get("compute_is_score", False):
            scoring["is_score"] = is_score

        add_params = {
            "scoring": scoring,
            "n_jobs": params.get("n_jobs", 1),
            "refit": params.get('refit', "accuracy"),
            "cv": params.get('cv', StratifiedKFold(n_splits=10, shuffle=True, random_state=42)),
            "verbose": params.get("verbose", 10),
            "pre_dispatch":params.get("pre_dispatch", "2*n_jobs"),
            "error_score": params.get("error_score", np.nan),
            "return_train_score": params.get("return_train_score", True),
        }

        for key, value in add_params.items():
            setattr(self.research.get_research(), key, value)

    @staticmethod
    def _setup_score():
        """
        Create default scoring dictionary for model evaluation.

        Parameters
        ----------
        compute_is_score : bool, optional
            If ``True``, include custom ``is_score`` metric.

        Returns
        -------
        dict
            Mapping from score name to scorer callable.
        """
        metric_average = "macro"

        score_dict = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average=metric_average, zero_division=0),
            "recall": make_scorer(recall_score, average=metric_average, zero_division=0),
            "f1": make_scorer(f1_score, average=metric_average, zero_division=0),
            "specificity": make_scorer(specificity_score, average=metric_average),
            "sensitivity": make_scorer(sensitivity_score, average=metric_average),
        }

        return score_dict

    @measure_time()
    def fit(self, X, y=None, groups=None):
        """
        Fit configured search strategy on provided data.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like, optional
            Target labels. Required for fitting.
        groups : array-like, optional
            Group labels for grouped CV and patient-level scoring.

        Returns
        -------
        object
            Fit result returned by underlying strategy.

        Raises
        ------
        Exception
            If estimator is not set.
        ValueError
            If ``y`` is missing or ``groups`` length mismatches ``X``.
        """
        if self.estimator is None:
            raise Exception("Estimator not set for RamanSearch.")

        if y is None:
            raise ValueError("Target 'y' is required for fitting.")

        if groups is not None and len(groups) != len(X):
            raise ValueError("Length of 'groups' must match number of samples in X.")

        if groups is not None:
            self.research.get_research().scoring['patient_accuracy'] = partial(patient_accuracy, groups=groups)

        self._fitted = True
        return self.research.fit_research(X, y, groups=groups)

    def predict(self, X):
        """
        Predict labels with the fitted best estimator.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        array-like
            Predicted labels.
        """
        return self.research.get_research().predict(X)

    def get_best_params(self):
        """
        Return best parameter set found during search.

        Returns
        -------
        dict
            Best parameter mapping.
        """
        return self.research.get_research().best_params_

    def get_best_score(self):
        """
        Return best score achieved during search.

        Returns
        -------
        float
            Best cross-validation score.
        """
        return self.research.get_research().best_score_

    def get_cv_results(self, file_path=None, **kwargs):
        """
        Return CV results or write them to CSV.

        Parameters
        ----------
        file_path : str, optional
            Output CSV path. If ``None``, returns results in-memory.

        Returns
        -------
        dict or None
            CV results dictionary when ``file_path`` is ``None``; otherwise
            writes to disk and returns ``None``.
        """
        if not self._fitted:
            raise Exception("RamanSearch is not fitted yet.")

        cv_res = pd.DataFrame(self.research.get_cv_results())

        if kwargs.get("return_split_scores", False):
            cv_res = cv_res.loc[:, cv_res.columns.str.startswith("split")]

        if kwargs.get("return_combined_params", False):
            pipeline_steps = self.estimator.get_steps()
            for i, step in enumerate(pipeline_steps):
                step_names = self._getPreprocessingString(cv_res, step)
                cv_res.insert(i, step, step_names)

            cv_res = cv_res.drop(columns=cv_res.filter(regex="^param").columns)

        if kwargs.get("round_values", False):
            # Round numerical columns
            numeric_cols = cv_res.select_dtypes(include=[np.number]).columns
            cv_res[numeric_cols] = cv_res[numeric_cols].round(4)

        if file_path is None:
            return cv_res
        cv_res.to_csv(file_path, index=False)


    def _getPreprocessingString(self, _dataframe, preprocessing):
        df_cat = _dataframe.filter(regex=f"^param_{preprocessing}")
        if len(df_cat.columns) == 0:
            df_cat[preprocessing.replace("param_", "")] = ""
            return df_cat[preprocessing.replace("param_", "")]

        # Get each candidate for the preprocessing
        df_cat_unique = df_cat.drop_duplicates()

        # Get the names of the algorithms
        algorithms_string = df_cat_unique.apply(self._getAlgorithmName, axis=1,
                                                preprocessing_name=f"{preprocessing}").to_dict()

        columns = df_cat.columns.values.tolist()
        # Group by the columns and get the indexes
        gr = df_cat.groupby(columns, dropna=False, observed=True).apply(lambda x: x.index.tolist())

        new_column_name = preprocessing.replace("param_", "")
        for index, row in gr.items():
            df_cat.loc[row, new_column_name] = algorithms_string[row[0]]

        return df_cat[new_column_name]

    def _getAlgorithmName(self, row, preprocessing_name):
        row = row.dropna()
        if row.empty:
            return "None"

        row.index = [x.replace(f"param_{preprocessing_name}__", "") for x in row.index]
        hyperparameters_string = ','.join([f'{index}={value}' for index, value in row.astype(str).items() if index != 'algorithm'])
        algorithms_name = str(row["algorithm"]).replace("()", "")

        if len(hyperparameters_string) == 0:
            final_string = f"{algorithms_name}"
        elif "=" in algorithms_name:
            algorithms_name = algorithms_name.replace(')', '')
            final_string = ','.join([algorithms_name, hyperparameters_string]) + ")"
        else:
            final_string = f"{algorithms_name}({hyperparameters_string})"
        return final_string
