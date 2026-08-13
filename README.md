# Ramantune

**Ramantune** is a Python library designed to streamline the development of machine learning models for Raman spectroscopy data. It enables co-optimization of preprocessing pipelines and classification algorithms within a unified framework.

## Features

- **Unified Pipeline**: Integrates spectral preprocessing with machine learning workflows
- **Hyperparameter Optimization**: Automatically explore combinations of preprocessing parameters and model hyperparameters
- **Parameter Validation**: Built-in validation ensures you specify only valid parameters for each algorithm
- **Flexible Architecture**: Easily construct configurable pipelines tailored to Raman spectroscopy data
- **Reproducible Experiments**: Robust cross-validation strategies support reproducible research

## Installation

```bash
pip install ramantune
```

## Quick Start

```python
from ramantune.pipeline import RamanPipeline
from ramantune.search import RamanSearch
from ramantune.search.search_space import (
    DenoiserSpace, BaselineSpace, NormalizerSpace,
    FeatureSelectionSpace, ClassifierSpace
)
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

# Prepare your data
X_train, y_train = ...  # your Raman spectroscopy data

# Define search space
param_grid = {
    'denoising': [DenoiserSpace('savgol', {'window_length': [7]})],
    'baseline': [BaselineSpace('imodpoly', {'poly_order': [3]})],
    'normalize': [NormalizerSpace('auc')],
    'feature_selection': [FeatureSelectionSpace(PCA(), {'n_components': [0.99]})],
    'classifier': [ClassifierSpace(SVC(), {'C': [1, 10]})],
}

# Create pipeline and search
pipeline = RamanPipeline()
search = RamanSearch(
    estimator=pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    compute_is_score=True
)

# Fit and retrieve results
search.fit(X_train, y_train)
results = search.get_cv_results()
print(f"Best parameters: {search.get_best_params()}")
print(f"Best score: {search.get_best_score()}")
```

## Why Ramantune?

In Raman-based classification tasks, preprocessing is critical. Decisions about baseline correction, smoothing, normalization, and spectral range selection strongly influence model performance. Ramantune addresses the challenge of jointly optimizing preprocessing and model hyperparameters by:

1. **Treating preprocessing and classification as a unified problem** rather than separate steps
2. **Validating parameters early** with helpful error messages
3. **Supporting scikit-learn compatible workflows** for seamless integration
4. **Facilitating reproducible research** with built-in cross-validation and results tracking

## Documentation

Full documentation is available at: [https://github.com/InnocenteSimone/ramantune](https://github.com/InnocenteSimone/ramantune)

Key topics:
- [Search Space Configuration](doc/search_space.rst) - Define parameter grids with automatic validation
- [Usage Guide](doc/usage.rst) - Complete examples and troubleshooting
- [API Reference](doc/ramanpipeline.rst) - Detailed API documentation

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.