from typing import Collection, Mapping
import numpy as np
from sklearn.model_selection._search import (
    BaseSearchCV,
    ParameterGrid,
    _check_param_grid,
)
from sklearn.utils.validation import check_is_fitted
from cluster_optimizer.scorer import get_scorer



class ClusterOptimizer(BaseSearchCV):
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
    cv_results_ : dict of numpy (masked) ndarrays
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
        if isinstance(scoring, str):
            scoring = get_scorer(scoring)
        if isinstance(refit, str):
            refit = get_scorer(refit)
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=[(slice(None), slice(None))],
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=False,
        )
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        if isinstance(value, str):
            self._scoring = get_scorer(value)
        else:
            self._scoring = value

    @property
    def labels_(self):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.labels_
