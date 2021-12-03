# cluster-optimizer

## Installation

You can install this package with pip using the following command:

```
pip git+https://github.com/ndgigliotti/cluster-optimizer.git@main
```

## Purpose

This project provides a simple, Scikit-Learn-compatible, hyperparameter optimization tool for clustering. It's intended for situations where there is no need to predict clusters for new data points. Many clustering algorithms in Scikit-Learn are transductive, meaning that they are not designed to be applied to new observations. Even if using an inductive clustering algorithm like K-Means, you might not have any desire to predict clusters for new observations.

Since Scikit-Learn's `GridSearchCV` uses cross-validation, and is designed to optimize inductive machine learning algorithms, it is necessary to create an alternative tool.

## `ClusterOptimizer`

The `ClusterOptimizer` class is a hyperparameter search tool for optimizing clustering algorithms. It simply fits one model per hyperparameter combination and selects the best. It's a spin-off of `GridSearchCV`, and uses the same Scikit-Learn machinery under the hood. The only difference is that it doesn't use cross-validation and is designed to work with special clustering scorers. It's optional to provide a target variable, since clustering metrics such as silhouette, Calinski-Harabasz, and Davies-Bouldin are designed for unsupervised clustering.

Currently, the `ClusterOptimizer` interface is the same as `GridSearchCV`, which might be a bit counterintuitive. For example, it acquires the `cv_results_` attribute after fitting, even though no actual cross-validation is performed. Still, someone used to `GridSearchCV` will have little to complain about.

## Transductive Clustering Scorers

You can use `ClusterOptimizer` by passing the string name of a Scikit-Learn clustering metric, e.g. 'silhouette', 'calinski_harabasz', or 'rand_score' (the '_score' suffix is optional). You can also create a special scorer for transductive clustering using `scorer.make_scorer` on any score function with the signature `score_func(labels_true, labels_fit)` or `score_func(X, labels_fit)`.


### Recognized Scorer Names

Note that the 'score' suffix is always optional.

- 'silhouette_score'
- 'silhouette_score_euclidean'
- 'silhouette_score_cosine'
- 'davies_bouldin_score'
- 'calinski_harabasz_score'
- 'mutual_info_score'
- 'normalized_mutual_info_score'
- 'adjusted_mutual_info_score'
- 'rand_score'
- 'adjusted_rand_score'
- 'completeness_score'
- 'fowlkes_mallows_score'
- 'homogeneity_score'
- 'v_measure_score'

## Reliance on Scikit-Learn

I could have created a cluster optimizer from scratch, but it would've taken much more work and wouldn't have been as reliable. Plus, people are familiar with Scikit-Learn's search estimators. Nevertheless, in the future, it might be a good idea to create a new version of `BaseSearchCV` (which contains the Scikit-Learn search machinery) for transductive clustering purposes.

## Future Work

- [x] Finish writing docstrings.
- [ ] Improve interface to reflect lack of CV.
- [ ] Add multi-metric compatibility.
- [ ] Write automated tests.
- [ ] Trim down environment file.
- [x] Make `setup.py`.