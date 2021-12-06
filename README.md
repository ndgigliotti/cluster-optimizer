# cluster-optimizer

## Installation

You can install this package with pip using the following command:

```
pip git+https://github.com/ndgigliotti/cluster-optimizer.git@main
```

## Purpose

This project provides a simple, Scikit-Learn-compatible, hyperparameter optimization tool for clustering. It's intended for situations where predicting clusters for new data points is a low priority. Many clustering algorithms in Scikit-Learn are **transductive**, meaning that they are not designed to be applied to new observations. Even if using an **inductive** clustering algorithm like K-Means, you might not have any desire to predict clusters for new observations. Or, even if you do have such a desire, prediction might be a lower priority than finding the best clusters in the training data.

Since Scikit-Learn's `GridSearchCV` uses cross-validation, and is designed to optimize inductive machine learning models, an alternative tool is necessary.

## `ClusterOptimizer`

`ClusterOptimizer` is a hyperparameter search tool for optimizing clustering models. It simply fits one model per hyperparameter combination and selects the best. It's a spin-off of `GridSearchCV`, and uses the same Scikit-Learn machinery under the hood. The only difference is that it doesn't use cross-validation and is designed to work with special clustering scorers. It's not always necessary to provide a target variable, since clustering metrics such as silhouette, Calinski-Harabasz, and Davies-Bouldin are designed for unsupervised clustering.

## Transductive Clustering Scorers

You can use `ClusterOptimizer` by passing the string name of a Scikit-Learn clustering metric, e.g. 'silhouette', 'calinski_harabasz', or 'rand_score' (the '_score' suffix is optional). You can also create a special scorer for transductive clustering using `scorer.make_scorer` on any score function with the signature `score_func(labels_true, labels_fit)` or `score_func(X, labels_fit)`.


### Recognized Scorer Names

Note that the '_score' suffix is always optional.

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

## Caveats

### Comparing Clustering Algorithms

It's important to consider your dataset and goals before comparing clustering algorithms in a grid search. Just because one algorithm gets a higher score than another does not necessarily make it a better choice. Different clustering algorithms have [different benefits, drawbacks, and use cases.](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods)

### Appropriation of the `GridSearchCV` Interface

`ClusterOptimizer` uses the same machinery as `GridSearchCV` under the hood, and for now, has the same interface. This has some counterintuitive consequences, such as log messages mentioning "CV" and the results being stored in the `cv_results_` attribute. It should be easy to use for someone familiar with `GridSearchCV`, but the naming is hardly ideal.

### No Multi-Metric Scoring

Multi-metric scoring is currently not supported, but there are plans to add it in the future.

## Future Work

### Independence from `BaseSearchCV`

Currently `ClusterOptimizer` relies heavily on `BaseSearchCV` (which defines the Scikit-Learn search machinery). I could have created a cluster optimizer from scratch, but it would've taken much more work to achieve a high level of reliability. Nevertheless, in the future, it would be a good idea to create a new version of `BaseSearchCV` for transductive clustering purposes.

### To-Do

- [x] Finish writing docstrings.
- [x] Make `setup.py`.
- [ ] Write automated tests.
- [ ] Trim down environment file.
- [ ] Develop alternative to `BaseSearchCV`.
- [ ] Add multi-metric compatibility.
