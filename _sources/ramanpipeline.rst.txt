Raman Pipeline
==============

Introduction
------------
The ``RamanPipeline`` class is the core orchestrator of Ramantune's preprocessing 
and classification workflow. It manages the assembly of preprocessing steps 
(denoising, baseline correction, normalization) with feature selection and 
classification algorithms into a unified scikit-learn compatible pipeline.

The pipeline is designed to be flexible: you can include or exclude preprocessing 
steps as needed, and all steps can have their hyperparameters optimized during 
model selection through ``RamanSearch``.

Pipeline Architecture
---------------------
The standard Ramantune pipeline consists of:

1. **Spectral Data Conversion** (internal)
2. **Denoising** (optional) - Remove noise from spectral data
3. **Baseline Correction** (optional) - Remove baseline artifacts
4. **Normalization** (optional) - Normalize spectral intensity
5. **Data Conversion** (internal)
6. **Feature Selection** (optional) - Extract relevant features
7. **Classification** (required) - Predict class labels

Each step is a scikit-learn compatible transformer or estimator with configurable 
hyperparameters. Steps are included in the pipeline based on the parameter grid 
provided to ``RamanSearch``.

API Reference
-------------

.. automodule:: ramantune.pipeline.raman_pipeline
   :members:
