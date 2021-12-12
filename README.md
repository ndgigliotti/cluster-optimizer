# cluster-optimizer

## Installation

You can install this package with pip using the following command:

```
pip git+https://github.com/ndgigliotti/cluster-optimizer.git@main
```

## Purpose

This project provides a simple, Scikit-Learn-compatible, hyperparameter optimization tool for clustering. It's intended for situations where predicting clusters for new data points is a low priority. Many clustering algorithms in Scikit-Learn are **transductive**, meaning that they are not designed to be applied to new observations. Even if using an **inductive** clustering algorithm like K-Means, you might not have any desire to predict clusters for new observations. Or, even if you do have such a desire, prediction might be a lower priority than finding the best clusters in the training data.

Since Scikit-Learn's `GridSearchCV` uses cross-validation, and is designed to optimize inductive machine learning algorithms, it is necessary to create an alternative tool.

## `ClusterOptimizer`

The `ClusterOptimizer` class is a hyperparameter search tool for optimizing clustering algorithms. It simply fits one model per hyperparameter combination and selects the best. It's a spin-off of `GridSearchCV`, and the implementation is derived from Scikit-Learn. The only difference is that it doesn't use cross-validation and is designed to work with special clustering scorers. It's not always necessary to provide a target variable, since clustering metrics such as silhouette, Calinski-Harabasz, and Davies-Bouldin are designed for unsupervised clustering.

The interface is largely the same as `GridSearchCV`. One minor difference is that the search results are stored in the `results_` attribute, rather than `cv_results_`.

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

### Only Tested for Clustering

So far, `ClusterOptimizer` has only been tested for clustering. It may have other applications, and may even work out of the box with some other Scikit-Learn estimators. If it proves to have a lot of other uses, it may need to be renamed.

## Future Work

- [x] Write automated tests.
- [x] Develop alternative to `BaseSearchCV`.
- [x] Add multi-metric compatibility.
- [ ] Explore applications beyond clustering (e.g. LDA).
- [ ] Update docstrings taken from Scikit-Learn.
- [ ] Add more search types (e.g. randomized).

## Credits

Most of the credit goes to the developers of Scikit-Learn for the engineering behind the search estimators. It's not very hard to spam a bunch of models with different hyperparameters, but it's hard to do it in a robust way with wide compatibility.
