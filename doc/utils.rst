Utilities
=========

Introduction
------------
This document describes the utility extension points provided by
``ramantune`` and explains how to add new preprocessing operators that
integrate with pipelines and hyperparameter search. The utilities
covered here include the algorithm registry (``register_algorithm`` and
``default_registry``) and the ``RamanPipelineStep`` wrapper used to
expose callables as pipeline-compatible preprocessing steps.

Key points
----------
- The registry stores preprocessing algorithm classes grouped by logical
  categories (denoising, baseline, normalization, ...). Categories are
  defined as constants in ``ramantune.utils.config`` (for example
  ``NORMALIZE_STR``).
- To add a new preprocessing algorithm you implement a wrapper
  class (subclassing ``ramantune.custom.RamanPipelineStep``) and register
  it via ``ramantune.utils.register_algorithm`` with the chosen category
  and a unique name.
- Ensure the module that performs registration is imported at program
  start (for example by packaging the module or importing it from your
  application). Registration is executed at import time and adds the
  algorithm to the global ``default_registry``.

Preprocessing extensions
========================

Registering new preprocessing algorithms
----------------------------------------
This document explains how to add a new preprocessing operator to the
ramantune registry so it can be used transparently by pipelines and
search utilities.

The steps are simple:

1. Create a wrapper class that subclasses
   ``ramantune.custom.RamanPipelineStep`` and implements the processing
   function. The wrapper should call ``super().__init__`` with the callable
   that performs the transformation.
2. Register the implementation using ``ramantune.utils.register_algorithm``
   with the appropriate category constant (for example
   ``NORMALIZE_STR`` for normalization routines).
3. Make sure the module containing the registration is imported at runtime
   (for example by packaging it or importing it in your application startup)
   so the algorithm is available in the global registry.

After registration you can reference the algorithm by name in a
``param_grid`` (for example by using ``NormalizerSpace('snv')``) or resolve
it directly via the default registry.

Example: SNV normalization
--------------------------
The following example shows a minimal implementation of an SNV
(Standard Normal Variate) normalization operator and how to register it
under the normalization category.

.. code-block:: python

    from ramantune.custom import RamanPipelineStep
    from ramantune.utils import register_algorithm
    from ramantune.utils.config import NORMALIZE_STR

    # Create a class that performs SNV normalization
    @register_algorithm(category=NORMALIZE_STR, name="snv")
    class SNVNormalization(RamanPipelineStep):
        """SNV normalization wrapper.

        The callable should accept ``spectral_data`` and ``spectral_axis`` and
        return a tuple ``(processed_data, spectral_axis)``. Processed data
        must be compatible with the downstream transformers in the pipeline.
        """
        def __init__(self):
            super().__init__(self.snv_normalization)

        @staticmethod
        def snv_normalization(spectral_data, spectral_axis):
            """Apply Standard Normal Variate normalization.

            Parameters
            ----------
            spectral_data : array-like
                One-dimensional spectral intensities for a single sample.
            spectral_axis : array-like
                The corresponding spectral axis (wavenumbers).

            Returns
            -------
            tuple
                (normalized_spectral_data, spectral_axis)
            """
            mean, std = spectral_data.mean(), spectral_data.std()
            return (spectral_data - mean) / std, spectral_axis

Using the registered algorithm
------------------------------
Once registered, reference the algorithm name in a parameter grid. For
example, when building a search space you can use:

.. code-block:: python

    from ramantune.search.search_space.search_space import NormalizerSpace

    param_grid = {
        'normalize': [NormalizerSpace('snv')],
        # ... other categories
    }


API Reference
-------------

.. automodule:: ramantune.utils.registry
   :members:
