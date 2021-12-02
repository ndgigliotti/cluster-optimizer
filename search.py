import numpy as np
from sklearn.model_selection._search import (
    BaseSearchCV,
    ParameterGrid,
    _check_param_grid,
)
from sklearn.utils.validation import check_is_fitted
from .scorer import get_scorer



class ClusterOptimizer(BaseSearchCV):
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
        if scoring is None:
            scoring = "silhouette"
        if isinstance(refit, str):
            refit = get_scorer(refit)
        super().__init__(
            estimator=estimator,
            scoring=get_scorer(scoring),
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
        self._scoring = get_scorer(value)

    @property
    def labels_(self):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.labels_
