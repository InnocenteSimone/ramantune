import ramanspy.preprocessing as rpr
from ramantune.utils.config import DENOISING_STR, BASELINE_STR, NORMALIZE_STR
"""
Algorithm registry for Raman preprocessing components.

This module provides a central registry used to resolve preprocessing
algorithm classes by category and name. It also exposes a decorator helper
for declarative registration and a default global registry instance.
"""

def register_algorithm(category: str, name: str):
    """
    Register an algorithm class into the default registry.

    Parameters
    ----------
    category : str
        Logical category (e.g., denoising, baseline, normalization).
    name : str
        Unique key used to retrieve the algorithm in ``category``.

    Returns
    -------
    callable
        Class decorator that registers the decorated class and returns it.
    """
    def decorator(cls):
        default_registry.add_algorithm(category=category, name=name, algorithm_class=cls)
        return cls
    return decorator

class AlgorithmRegistry:
    """
        Store and resolve preprocessing algorithm classes by category.
        Notes
        -----
        The registry is initialized with built-in ``ramanspy.preprocessing``
        algorithms and can be extended at runtime.
    """
    def __init__(self):
        self._algorithms = {
            DENOISING_STR: {
                "whittaker": rpr.denoise.Whittaker,
                "savgol": rpr.denoise.SavGol,
                "kernel": rpr.denoise.Kernel,
                "gaussian": rpr.denoise.Gaussian,
            },
            BASELINE_STR: {
                "asls": rpr.baseline.ASLS,
                "iasls": rpr.baseline.IASLS,
                "airpls": rpr.baseline.AIRPLS,
                "arpls": rpr.baseline.ARPLS,
                "drpls": rpr.baseline.DRPLS,
                "poly": rpr.baseline.Poly,
                "modpoly": rpr.baseline.ModPoly,
                "goldindec": rpr.baseline.Goldindec,
                "irsqr": rpr.baseline.IRSQR,
                "cornercutting": rpr.baseline.CornerCutting,
                "imodpoly": rpr.baseline.IModPoly,
            },
            NORMALIZE_STR: {
                "vector": rpr.normalise.Vector,
                "minmax": rpr.normalise.MinMax,
                "max_intensity": rpr.normalise.MaxIntensity,
                "auc": rpr.normalise.AUC,
            },
        }

    def get_category(self, category: str):
        """
        Return all algorithms registered under a category.

        Parameters
        ----------
        category : str
            Category key.

        Returns
        -------
        dict
            Mapping from algorithm names to classes. Empty dict if missing.
        """
        return self._algorithms.get(category, {})

    def add_algorithm(self, category: str, name: str, algorithm_class):
        """
        Add or replace an algorithm class in a category.

        Parameters
        ----------
        category : str
            Category key.
        name : str
            Algorithm name.
        algorithm_class : type
            Class implementing the algorithm.
        """
        self._algorithms.setdefault(category, {})[name] = algorithm_class

    def get_algorithm(self, category: str, name: str):
        """
        Return a registered algorithm class.

        Parameters
        ----------
        category : str
            Category key.
        name : str
            Algorithm name in the category.

        Returns
        -------
        type
            Registered algorithm class.

        Raises
        ------
        KeyError
            If the category or algorithm name does not exist.
        """
        try:
            return self._algorithms[category][name]
        except KeyError as exc:
            raise KeyError(f"Algorithm '{name}' not found in category '{category}'") from exc

    def get_available_algorithms(self, category: str):
        """
        Return the list algorithm names available in a category.

        Parameters
        ----------
        category : str
            Category key.

        Returns
        -------
        list of str
            Available algorithm names. Empty list if category is missing.
        """
        return list(self._algorithms.get(category, {}).keys())

default_registry = AlgorithmRegistry()