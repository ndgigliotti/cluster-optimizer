from types import MappingProxyType
from typing import Iterable
from sklearn import metrics
from sklearn.metrics._scorer import _BaseScorer, _passthrough_scorer
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
        if isinstance(estimator, Pipeline):
            X = estimator[:-1].transform(X)
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
                "Use sorted(cluster_optimizer.scorer.SCORERS.keys()) "
                "to get valid options."
            )
    else:
        scorer = scoring
    return scorer


def check_scoring(estimator, scoring=None):
    """Determine scorer from user options.

    A TypeError will be thrown if the estimator cannot be scored.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    """
    if not hasattr(estimator, "fit"):
        raise TypeError(
            "estimator should be an estimator implementing "
            "'fit' method, %r was passed" % estimator
        )
    if isinstance(scoring, str):
        return get_scorer(scoring)
    elif callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("sklearn.metrics.")
            and not module.startswith("sklearn.metrics._scorer")
            and not module.startswith("sklearn.metrics.tests.")
        ):
            raise ValueError(
                "scoring value %r looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_scorer` to convert a metric "
                "to a scorer." % scoring
            )
        return get_scorer(scoring)
    elif scoring is None:
        if hasattr(estimator, "score"):
            return _passthrough_scorer
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not." % estimator
            )
    elif isinstance(scoring, Iterable):
        raise ValueError(
            "For evaluating multiple scores, use "
            "sklearn.model_selection.cross_validate instead. "
            "{0} was passed.".format(scoring)
        )
    else:
        raise ValueError(
            "scoring value should either be a callable, string or"
            " None. %r was passed" % scoring
        )


def check_multimetric_scoring(estimator, scoring):
    """Check the scoring parameter in cases when multiple metrics are allowed.

    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.

    scoring : list, tuple or dict
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        See :ref:`multimetric_grid_search` for an example.

    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    """
    err_msg_generic = (
        f"scoring is invalid (got {scoring!r}). Refer to the "
        "scoring glossary for details: "
        "https://scikit-learn.org/stable/glossary.html#term-scoring"
    )

    if isinstance(scoring, (list, tuple, set)):
        err_msg = (
            "The list/tuple elements must be unique " "strings of predefined scorers. "
        )
        invalid = False
        try:
            keys = set(scoring)
        except TypeError:
            invalid = True
        if invalid:
            raise ValueError(err_msg)

        if len(keys) != len(scoring):
            raise ValueError(
                f"{err_msg} Duplicate elements were found in"
                f" the given list. {scoring!r}"
            )
        elif len(keys) > 0:
            if not all(isinstance(k, str) for k in keys):
                if any(callable(k) for k in keys):
                    raise ValueError(
                        f"{err_msg} One or more of the elements "
                        "were callables. Use a dict of score "
                        "name mapped to the scorer callable. "
                        f"Got {scoring!r}"
                    )
                else:
                    raise ValueError(
                        f"{err_msg} Non-string types were found "
                        f"in the given list. Got {scoring!r}"
                    )
            scorers = {
                scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring
            }
        else:
            raise ValueError(f"{err_msg} Empty list was given. {scoring!r}")

    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError(
                "Non-string types were found in the keys of "
                f"the given dict. scoring={scoring!r}"
            )
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")
        scorers = {
            key: check_scoring(estimator, scoring=scorer)
            for key, scorer in scoring.items()
        }
    else:
        raise ValueError(err_msg_generic)
    return scorers
