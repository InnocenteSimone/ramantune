from ramantune.pipeline.pipeline_builder import PipelineBuilder
from ramantune.utils import (
    DENOISING_STR,
    BASELINE_STR,
    NORMALIZE_STR,
    FEATURE_SELECTION_STR,
    CLASSIFIER_STR,
    TO_SPECTRA_STR,
    TO_VALUES_STR)

from .transformers import *
from ramantune.utils import default_registry

class RamanPipeline:
    """Build and manage a Raman analysis pipeline.

    Parameters
    ----------
    registry : object, optional
        Registry-like object exposing ``get_category(name)`` to retrieve
        preprocessing operator registries by category. By default,
        ``default_registry`` is used.

    Attributes
    ----------
    pipeline : object or None
        Built pipeline instance returned by ``PipelineBuilder.build()``.
        It is initialized during object creation.
    registry : object
        Registry used to resolve preprocessing components.
    """

    def __init__(self, registry=default_registry):
        """Initialize and build the default full pipeline.

        Parameters
        ----------
        registry : object, optional
            Registry used to resolve available preprocessing methods.
        """
        self.__pipeline_builder = PipelineBuilder()
        self.pipeline = None
        self.registry = registry

        self._build_all_pipeline()

    def _build_all_pipeline(self):
        """Build the default pipeline with all preprocessing categories.

        Notes
        -----
        The default pipeline always includes denoising, baseline correction,
        normalization, feature selection, and classification in a fixed order.
        """
        self.__pipeline_builder.add_step(TO_SPECTRA_STR, SpectraTransformer())
        self.__pipeline_builder.add_step(DENOISING_STR, RamanPreprocessingTransformer(category=DENOISING_STR,
                                                                                      registry=self.registry.get_category(DENOISING_STR)))
        self.__pipeline_builder.add_step(BASELINE_STR, RamanPreprocessingTransformer(category=BASELINE_STR,
                                                                                     registry=self.registry.get_category(BASELINE_STR)))
        self.__pipeline_builder.add_step(NORMALIZE_STR, RamanPreprocessingTransformer(category=NORMALIZE_STR,
                                                                                      registry=self.registry.get_category(NORMALIZE_STR)))
        self.__pipeline_builder.add_step(TO_VALUES_STR, InverseSpectraTransformer())
        self.__pipeline_builder.add_step(FEATURE_SELECTION_STR, FeatureSelectionTransformer())
        self.__pipeline_builder.add_step(CLASSIFIER_STR, ModelTransformer())
        self.pipeline = self.__pipeline_builder.build()

    def get_steps(self):
        """Return names of active tunable steps in the build pipeline.

        Returns
        -------
        list of str
            Step names excluding internal conversion stages
            (``TO_SPECTRA_STR`` and ``TO_VALUES_STR``). Returns an empty list
            when no pipeline is currently built.
        """
        return [el[0] for el in self.pipeline.steps if el[0] not in [TO_SPECTRA_STR, TO_VALUES_STR]] if self.pipeline else []

    def build_pipeline(self, param_grid):
        """Build a pipeline from a provided hyperparameter grid.

        Parameters
        ----------
        param_grid : dict
            Hyperparameter grid keyed by step/category names. Preprocessing
            categories are included only when present in this mapping.

        Notes
        -----
        Classification is always included. Feature selection is optional and
        enabled only when ``FEATURE_SELECTION_STR`` exists in ``param_grid``.
        """
        self.__pipeline_builder.reset()
        self.__pipeline_builder.add_step(TO_SPECTRA_STR, SpectraTransformer())
        for category in [DENOISING_STR, BASELINE_STR, NORMALIZE_STR]:
            if category in param_grid:
                self.__pipeline_builder.add_step(category, RamanPreprocessingTransformer(category=category,
                                                                                         registry=self.registry.get_category(category)))
        self.__pipeline_builder.add_step(TO_VALUES_STR, InverseSpectraTransformer())

        if FEATURE_SELECTION_STR in param_grid:
            self.__pipeline_builder.add_step(FEATURE_SELECTION_STR, FeatureSelectionTransformer())

        self.__pipeline_builder.add_step(CLASSIFIER_STR, ModelTransformer())
        self.pipeline = self.__pipeline_builder.build()

    def get_pipeline(self):
        """Return the currently built pipeline instance.

        Returns
        -------
        object or None
            Pipeline instance produced by ``PipelineBuilder``, or ``None`` if
            not built.
        """
        return self.pipeline

