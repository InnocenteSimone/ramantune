from sklearn.model_selection import RandomizedSearchCV
from .context.abstract import ResearchStrategy
from ramantune.utils import ALGORITHM_STR, CLASSIFIER_STR
from scipy.stats._distn_infrastructure import rv_frozen
from itertools import product as iterproduct

class RandomSearchStrategy(ResearchStrategy, RandomizedSearchCV):

    def __init__(self):
        RandomizedSearchCV.__init__(self, estimator=None, param_distributions={})
        ResearchStrategy.__init__(self)

    def expand_distributions(self, param_dict):
        """
        Splits any key whose value is a list of distributions
        into separate dicts — one per distribution.
        """

        def is_dist(v):
            return hasattr(v, 'rvs')  # any scipy distribution has .rvs()

        # Find keys that have a list of distributions
        dist_keys = {k: v for k, v in param_dict.items()
                     if isinstance(v, list) and any(is_dist(i) for i in v)}

        static_keys = {k: v for k, v in param_dict.items()
                       if k not in dist_keys}

        if not dist_keys:
            return [param_dict]

        # Expand: one dict per combination
        all_combos = iterproduct(*[[(k, d) for d in dists]
                                   for k, dists in dist_keys.items()])

        result = []
        for combo in all_combos:
            new_dict = {**static_keys}
            for k, dist in combo:
                new_dict[k] = dist  # single distribution, not a list
            result.append(new_dict)

        return result


    def get_classifier_params(self, params):
        for d in params:
            to_erase = []
            for hyperparameter, values in d.items():
                if hyperparameter == f"{CLASSIFIER_STR}__{ALGORITHM_STR}":
                    continue

                # If the number of value for a particular hyperparameter is 1, setup directly into the classifier
                if (len(values) == 1) and not isinstance(values[0], rv_frozen):
                    d[f"{CLASSIFIER_STR}__{ALGORITHM_STR}"] = d[f"{CLASSIFIER_STR}__{ALGORITHM_STR}"].set_params(
                        **{hyperparameter.replace(f"{CLASSIFIER_STR}__", ""): values[0]}
                    )
                    to_erase.append(hyperparameter)

            for el in to_erase:
                d.pop(el)

        return self.expand_distributions(params[0])

    def get_preprocessing_params(self, params):
        return self.expand_distributions(params[0])


    def setup_search_space(self, params):
        self.param_distributions = params

    def fit_strategy(self, X, y=None, groups=None, **fit_params):
        return self.fit(X, y, groups=groups, **fit_params)

    def fit(self, X, y=None, groups=None, **fit_params):
        super(RandomizedSearchCV, self).fit(X, y, groups=groups, **fit_params)
        return self

    def get_cv_results(self):
        return self.cv_results_