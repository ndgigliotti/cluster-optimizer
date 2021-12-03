from types import MappingProxyType
from sklearn import metrics
from sklearn.metrics._scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_consistent_length, check_is_fitted


def _get_labels(estimator):
    """Gets the cluster labels from an estimator or pipeline."""
    if isinstance(estimator, Pipeline):
        check_is_fitted(estimator._final_estimator, ["labels_"])
        labels = estimator._final_estimator.labels_
    else:
        check_is_fitted(estimator, ["labels_"])
        labels = estimator.labels_
    return labels


def _remove_noise_cluster(*arrays, labels, noise_label=-1):
    """
    Removes the noise cluster found in `labels` (if any) from all `arrays`.
    
    This function is currently unused, and may be removed in the future.
    Initially, it seemed like a good idea to remove the noise "cluster" when
    scoring algorithms like DBSCAN or HDBSCAN. However, this proved to break the
    optimizer. If the noise cluster is removed, these two algorithms can attain
    very high scores while classifying 99% of the points as noise. Such misleadingly
    high scores are robustly selected for on some datasets. Furthermore,
    retaining the noise "cluster" does not seem to cause problems.

    """
    is_noise = labels == noise_label
    arrays = list(arrays)
    for i, arr in enumerate(arrays):
        arrays[i] = arr[~is_noise].copy()
    check_consistent_length(*arrays)
    return tuple(arrays)


class _LabelScorerSupervised(_BaseScorer):
    def _score(self, estimator, X, labels_true):
        """Evaluate estimator labels relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Does nothing, since estimator should already have `labels_`.
            Here for API compatability.

        labels_true : array-like
            Ground truth target values for cluster labels.

        Returns
        -------
        score : float
            Score function applied to cluster labels.
        """

        labels = _get_labels(estimator)
        return self._sign * self._score_func(labels_true, labels, **self._kwargs)

    def __call__(self, estimator, X, labels_true):
        """Evaluate estimator labels relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Does nothing, since estimator should already have `labels_`.
            Here for API compatability.

        labels_true : array-like
            Ground truth target values for cluster labels.

        Returns
        -------
        score : float
            Score function applied to cluster labels.
        """
        return self._score(
            estimator,
            X,
            labels_true,
        )


class _LabelScorerUnsupervised(_BaseScorer):
    def _score(self, estimator, X, labels_true=None):
        """Evaluate cluster labels on X.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Data that will be used to evaluate cluster labels.

        labels_true: array-like
            Does nothing. Here for API compatability.

        Returns
        -------
        score : float
            Score function applied to cluster labels.
        """
        labels = _get_labels(estimator)
        return self._sign * self._score_func(X, labels, **self._kwargs)

    def __call__(self, estimator, X, labels_true=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have `labels_` attribute.

        X : {array-like, sparse matrix}
            Data that will be used to evaluate cluster labels.

        labels_true: array-like
            Does nothing. Here for API compatability.

        Returns
        -------
        score : float
            Score function applied cluster labels.
        """
        return self._score(
            estimator,
            X,
        )


def make_scorer(
    score_func,
    *,
    ground_truth=True,
    greater_is_better=True,
    **kwargs,
):
    """Make a clustering scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in
    :class:`~cluster_opt.ClusterOptimizer`
    It takes a score function, such as :func:`~sklearn.metrics.silhouette_score`,
    :func:`~sklearn.metrics.mutual_info_score`, or
    :func:`~sklearn.metrics.adjusted_rand_index`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    ground_truth : bool, default=True
        Whether score_func uses ground truth labels.

    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    if ground_truth:
        cls = _LabelScorerSupervised
    else:
        cls = _LabelScorerUnsupervised
    return cls(score_func, sign, kwargs)


SCORERS = {
    "silhouette_score": make_scorer(metrics.silhouette_score, ground_truth=False),
    "silhouette_score_euclidean": make_scorer(
        metrics.silhouette_score, ground_truth=False
    ),
    "silhouette_score_cosine": make_scorer(
        metrics.silhouette_score, ground_truth=False, metric="cosine"
    ),
    "davies_bouldin_score": make_scorer(
        metrics.davies_bouldin_score, greater_is_better=False, ground_truth=False
    ),
    "calinski_harabasz_score": make_scorer(
        metrics.calinski_harabasz_score, ground_truth=False
    ),
    "mutual_info_score": make_scorer(metrics.mutual_info_score),
    "normalized_mutual_info_score": make_scorer(metrics.normalized_mutual_info_score),
    "adjusted_mutual_info_score": make_scorer(metrics.adjusted_mutual_info_score),
    "rand_score": make_scorer(metrics.rand_score),
    "adjusted_rand_score": make_scorer(metrics.adjusted_rand_score),
    "completeness_score": make_scorer(metrics.completeness_score),
    "fowlkes_mallows_score": make_scorer(metrics.fowlkes_mallows_score),
    "homogeneity_score": make_scorer(metrics.homogeneity_score),
    "v_measure_score": make_scorer(metrics.v_measure_score),
}
SCORERS.update({k.replace("_score", ""): v for k, v in SCORERS.items()})
SCORERS = MappingProxyType(SCORERS)


def get_scorer(scoring):
    """Get a clustering scorer from string.

    Parameters
    ----------
    scoring : str or callable
        Scoring method as string. If callable it is returned as is.

    Returns
    -------
    scorer : callable
        The scorer.
    """
    if isinstance(scoring, str):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            raise ValueError(
                f"'{scoring}' is not a valid scoring value. "
                "Use sorted(cluster_opt.SCORERS.keys()) "
                "to get valid options."
            )
    else:
        scorer = scoring
    return scorer