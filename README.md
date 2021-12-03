# cluster-optimizer

## Purpose

This project provides a simple, Scikit-Learn-compatible, hyperparameter optimization tool for clustering. It's intended for situations where there is no need to predict clusters for new data points. Many clustering algorithms in Scikit-Learn are transductive, meaning that they are not designed to be applied to new observations. Even if using an inductive clustering algorithm like K-Means, you might not have any desire to predict clusters for new observations.

Since Scikit-Learn's `GridSearchCV` uses cross-validation, and is designed to optimize inductive machine learning algorithms, it is necessary to create an alternative tool.

## `ClusterOptimizer`

The `ClusterOptimizer` class is a hyperparameter search tool for optimizing clustering algorithms. It simply fits one model per hyperparameter combination and selects the best. It's a spin-off of `GridSearchCV`, and uses the same Scikit-Learn machinery under the hood. The only difference is that it doesn't use cross-validation and is designed to work with special clustering scorers. It's optional to provide a target variable, since clustering metrics such as silhouette, Calinski-Harabasz, and Davies-Bouldin are designed for unsupervised clustering.

Currently, the `ClusterOptimizer` interface is the same as `GridSearchCV`, which might be a bit counterintuitive. For example, it acquires the `cv_results_` attribute after fitting, even though no actual cross-validation is performed. Still, someone used to `GridSearchCV` will have little to complain about.

## Transductive Clustering Scorers

You can use `ClusterOptimizer` by passing the string name of a Scikit-Learn clustering metric, e.g. 'silhouette', 'calinski_harabasz', or 'rand_score' (the '_score' suffix is optional). You can also create a special scorer for transductive clustering using `scorer.make_scorer` on any of Scikit-Learn's clustering metrics.

The default metric for `ClusterOptimizer` is silhouette score with Euclidean distance. Currently 'silhouette_cosine' is the only recognized string for specifying non-Euclidean distance, but more will be added soon.

## Reliance on Scikit-Learn

I could have created a cluster optimizer from scratch, but it would've taken much more work and wouldn't have been as feature-packed and reliable. Plus, people are familiar with Scikit-Learn's search estimators.

The only downsides of relying on `BaseSearchCV` from Scikit-Learn are superficial. Some attributes are weirdly named, the log messages mention "CV" and "folds" unnecessarily, `cv_results_` contains weirdly named fields. All of these things are minor, and most of them can be fixed.

## Future Work

- [ ] Finish writing docstrings.
- [ ] Improve interface to reflect lack of CV.
- [ ] Add multi-metric compatibility.
- [ ] Write automated tests.
- [ ] Trim down environment file.
- [ ] Make `setup.py`.