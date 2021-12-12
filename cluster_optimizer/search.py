import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from contextlib import suppress
from functools import partial
from traceback import format_exc

import joblib
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.model_selection._search import ParameterGrid, _check_param_grid
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _insert_error_scores,
    _normalize_score_results,
)
from sklearn.pipeline import Pipeline
from sklearn.utils._tags import _safe_tags
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import _check_fit_params, check_is_fitted

from cluster_optimizer.scorer import (
    _get_labels,
    check_multimetric_scoring,
    check_scoring,
)


def _score(estimator, X, y, scorer, error_score="raise"):
    """Compute the score(s) of an estimator on a given data set.

    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        scorers = scorer
    else:
        scorers = {0: scorer}
    try:
        if y is None:
            scores = {name: scorers[name](estimator, X) for name in scorers}
        else:
            scores = {name: scorers[name](estimator, X, y) for name in scorers}
    except Exception:
        if error_score == "raise":
            raise
        else:
            scores = {name: error_score for name in scorers}
            warnings.warn(
                f"Scoring failed. The score for these parameters"
                f" will be set to {error_score}. Details: \n"
                f"{format_exc()}",
                UserWarning,
            )

    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"

    for name, score in scores.items():
        if hasattr(score, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                score = score.item()
        if not isinstance(score, numbers.Number):
            raise ValueError(error_msg % (score, type(score), name))
        scores[name] = score
    if len(scores) == 1:
        scores = list(scores.values())[0]
    return scores


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    verbose,
    parameters,
    fit_params,
    return_parameters=False,
    return_n_samples=False,
    return_times=False,
    return_estimator=False,
    candidate_progress=None,
    error_score=np.nan,
):

    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_failed : bool
            The estimator failed to fit.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if candidate_progress and verbose > 9:
            progress_msg += f"{candidate_progress[0]+1}/" f"{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CANDIDATE {progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    # X_train, y_train = _safe_split(estimator, X, y, train)
    # X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y is None:
            estimator.fit(X, **fit_params)
        else:
            estimator.fit(X, y, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                scores = {name: error_score for name in scorer}
            else:
                scores = error_score
            warnings.warn(
                "Estimator fit failed. The score for these parameters"
                " will be set to %f. Details: \n%s" % (error_score, format_exc()),
                FitFailedWarning,
            )
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        scores = _score(estimator, X, y, scorer, error_score)
        score_time = time.time() - start_time - fit_time

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CANDIDATE {progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2 and isinstance(scores, dict):
            for scorer_name in sorted(scores):
                result_msg += f" {scorer_name}: ("
                result_msg += f"score={scores[scorer_name]:.3f})"
        result_msg += f" total time={joblib.logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["scores"] = scores
    if return_n_samples:
        if isinstance(estimator, Pipeline):
            result["n_samples"] = estimator[:-1].transform(X).shape[0]
        else:
            result["n_samples"] = X.shape[0]
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


class BaseSearch(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for hyperparameter search."""

    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
    ):
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def _more_tags(self):
        # allows cross-validation to see 'precomputed' metrics
        return {
            "pairwise": _safe_tags(self.estimator, "pairwise"),
            "_xfail_checks": {
                "check_supervised_y_2d": "DataConversionWarning not caught"
            },
        }

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        self._check_is_fitted("score")
        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the estimator doesn't provide one %s" % self.best_estimator_
            )
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y)

        # callable
        score = self.scorer_(self.best_estimator_, X, y)
        if self.multimetric_:
            score = score[self.refit]
        return score

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def score_samples(self, X):
        """Call score_samples on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``score_samples``.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
        """
        self._check_is_fitted("score_samples")
        return self.best_estimator_.score_samples(X)

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError(
                "This %s instance was initialized "
                "with refit=False. %s is "
                "available only after refitting on the best "
                "parameters. You can refit an estimator "
                "manually using the ``best_params_`` "
                "attribute" % (type(self).__name__, method_name)
            )
        else:
            check_is_fitted(self)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("predict")
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("predict_proba")
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("predict_log_proba")
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("decision_function")
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("transform")
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("inverse_transform")
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the search estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    def _run_search(self, evaluate_candidates):
        """Repeatedly calls `evaluate_candidates` to conduct a search.

        This method, implemented in sub-classes, makes it possible to
        customize the the scheduling of evaluations: GridSearchCV and
        RandomizedSearchCV schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearchCV abstract base class.
        For example, Successive Halving is implemented by calling
        `evaluate_candidates` multiples times (once per iteration of the SH
        process), each time passing a different set of candidates with `X`
        and `y` of increasing sizes.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits, or
                  evaluate candidates on subsampled data (as done in the
                  SucessiveHaling estimators). By default, the original `cv`
                  parameter is used, and it is available as a private
                  `_checked_cv_orig` attribute.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`

            It returns a dict of all results so far, formatted like
            ``cv_results_``.

            Important note (relevant whether the default cv is used or not):
            in randomized splitters, and unless the random_state parameter of
            cv was set to an int, calling cv.split() multiple times will
            yield different splits. Since cv.split() is called in
            evaluate_candidates, this means that candidates will be evaluated
            on different splits each time evaluate_candidates is called. This
            might be a methodological issue depending on the search strategy
            that you're implementing. To prevent randomized splitters from
            being used, you may use _split._yields_constant_splits()

        Examples
        --------

        ::

            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
        raise NotImplementedError("_run_search not implemented.")

    def _check_refit_for_multimetric(self, scores):
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if (
            self.refit is not False
            and not valid_refit_dict
            and not callable(self.refit)
        ):
            raise ValueError(multimetric_refit_msg)

    def fit(self, X, y=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        # estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        # X, y = indexable(X, y)
        fit_params = _check_fit_params(X, fit_params)

        base_estimator = clone(self.estimator)

        fit_and_score = delayed(
            partial(
                _fit_and_score,
                scorer=scorers,
                fit_params=fit_params,
                return_n_samples=True,
                return_times=True,
                return_parameters=False,
                error_score=self.error_score,
                verbose=self.verbose,
            )
        )

        results = {}
        with joblib.Parallel(
            n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch
        ) as workers:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, more_results=None):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(f"Fitting {n_candidates} candidates.")

                out = workers(
                    fit_and_score(
                        clone(base_estimator),
                        X,
                        y,
                        parameters=parameters,
                        candidate_progress=(cand_idx, n_candidates),
                    )
                    for cand_idx, parameters in enumerate(candidate_params)
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. Were there no candidates?"
                    )
                elif len(out) != n_candidates:
                    raise ValueError(f"Received fewer results than candidates.")

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_score = all_out[0]["scores"]
            self.multimetric_ = isinstance(first_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError("best_index_ returned is not an integer")
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError("best_index_ index out of range")
            else:
                self.best_index_ = results["rank_%s" % refit_metric].argmin()
                self.best_score_ = results[refit_metric][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.results_ = results

        return self

    def _format_results(self, candidate_params, out, more_results=None):
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)

        results = dict(more_results or {})

        def _store(key_name, array, rank=False):
            """A small helper to store the scores/times to the results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64)

            results[key_name] = array

            if np.any(~np.isfinite(array)):
                warnings.warn(
                    f"One or more of the {key_name} scores " f"are non-finite: {array}",
                    category=UserWarning,
                )

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array, method="min"), dtype=np.int32
                )

        _store("fit_time", out["fit_time"])
        _store("score_time", out["score_time"])
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        scores_dict = _normalize_score_results(out["scores"])

        for scorer_name in scores_dict:
            _store(
                scorer_name,
                scores_dict[scorer_name],
                rank=True,
            )

        return results


class ClusterOptimizer(BaseSearch):
    """Exhaustive search over specified parameter values for a clustering estimator.

    ClusterOptimizer implements a `fit` and a `score` method. It attains the
    `labels_` attribute after optimizing hyperparemeters if `refit=True`.
    It also implements "predict", "transform" and "inverse_transform"
    if they are implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by a simple grid-search over a parameter grid. There is no cross-validation;
    one fit is performed on the full data per entry in the grid.

    Parameters
    ----------
    estimator : clustering estimator.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : str, callable, default=None
        A single str or a callable to evaluate the fit on the data.

        NOTE that when using custom scorers, each scorer should return a single
        value. Consider using `scorer.make_scorer` on a function with the signature
        score_func(labels_true, labels_fit) or score_func(X, labels_fit).

        If None, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a `joblib.parallel_backend` context.
        ``-1`` means using all processors.

    pre_dispatch : int, or str, default=n_jobs
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Attributes
    ----------
    results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score on the full data. Not available
        if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations). Always
        1, since cross-validation is performed in name only. The one "split"
        is actually not a split at all, since the full training dataset is passed
        to the scorer.

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics. Currently not supported.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    scorer.make_scorer : Make a scorer from a performance metric.

    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
        )
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))

    @property
    def labels_(self):
        check_is_fitted(self, "best_estimator_")
        return _get_labels(self.best_estimator_)
