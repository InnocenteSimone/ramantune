Ramantune Documentation
=======================

**Ramantune** is a Python library designed to streamline the development of 
machine learning models for Raman spectroscopy data.

Overview
--------

In Raman-based classification tasks, preprocessing plays a critical role: 
choices such as baseline correction, smoothing, normalization, and spectral 
range selection can strongly influence model performance. However, these 
steps are often treated separately from the machine learning model, leading 
to suboptimal results.

Ramantune addresses this challenge by enabling the **co-optimization** of 
preprocessing pipelines and classification algorithms within a unified framework. 
The library integrates spectral preprocessing steps with machine learning 
workflows, allowing users to automatically explore combinations of preprocessing 
parameters and model hyperparameters during model selection.

Key Features
------------

- **Unified Framework**: Combines preprocessing and model selection into one pipeline
- **Automatic Validation**: Built-in parameter validation catches errors early
- **Flexible Configuration**: Define custom parameter grids for automated exploration
- **Scikit-learn Compatible**: Works seamlessly with sklearn's cross-validation and GridSearch
- **Extensible**: Easily add custom preprocessing algorithms
- **Reproducible**: Track and export all results for reproducible research

Getting Started
---------------
   See :doc:`usage` for complete examples.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   search_space
   ramanpipeline
   ramansearch
   utils
   usage


