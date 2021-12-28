import numpy as np
import pandas as pd
import pytest
from sklearn import cluster, datasets, decomposition
from sklearn import preprocessing as prep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from cluster_optimizer import ClusterOptimizer


@pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
def test_singular_metric():
    df, _ = datasets.load_iris(return_X_y=True, as_frame=True)
    df = pd.DataFrame(df)
    grid = {"eps": np.arange(0.25, 3.25, 0.25), "min_samples": [5, 20, 50]}
    search = ClusterOptimizer(
        cluster.DBSCAN(), grid, scoring="silhouette", error_score=-1
    )
    search.fit(df)
    check_is_fitted(
        search,
        [
            "results_",
            "best_score_",
            "best_index_",
            "best_estimator_",
            "best_params_",
            "labels_",
        ],
    )
    assert round(search.best_score_, 1) == 0.7
    assert search.best_params_["eps"] == 0.75
    assert search.best_params_["min_samples"] == 20
    assert sorted(search.results_.keys()) == sorted(
        [
            "fit_time",
            "param_eps",
            "param_min_samples",
            "params",
            "rank_score",
            "score",
            "score_time",
            "noise_ratio",
        ]
    )


@pytest.mark.filterwarnings(
    "ignore:Scoring failed",
    "ignore:One or more",
    "ignore:Noise ratio",
)
def test_multi_metric():
    df, _ = datasets.load_iris(return_X_y=True, as_frame=True)
    df = pd.DataFrame(df)
    grid = {"eps": np.arange(0.25, 3.25, 0.25), "min_samples": [5, 20, 50]}
    search = ClusterOptimizer(
        cluster.DBSCAN(),
        grid,
        scoring=["silhouette", "calinski_harabasz", "davies_bouldin_score"],
        refit="silhouette",
    )
    search.fit(df)
    check_is_fitted(
        search,
        [
            "results_",
            "best_score_",
            "best_index_",
            "best_estimator_",
            "best_params_",
            "labels_",
        ],
    )
    assert round(search.best_score_, 1) == 0.7
    assert search.best_params_["eps"] == 0.75
    assert search.best_params_["min_samples"] == 20
    assert sorted(search.results_.keys()) == sorted(
        [
            "fit_time",
            "param_eps",
            "param_min_samples",
            "params",
            "rank_silhouette",
            "rank_davies_bouldin_score",
            "davies_bouldin_score",
            "rank_calinski_harabasz",
            "calinski_harabasz",
            "silhouette",
            "score_time",
            "noise_ratio",
        ]
    )


@pytest.mark.filterwarnings("ignore:Scoring failed", "ignore:Noise ratio")
def test_pipeline():
    text = [
        pd.DataFrame.__doc__,
        pd.Series.__doc__,
        prep.Binarizer.__doc__,
        prep.MultiLabelBinarizer.__doc__,
        prep.OneHotEncoder.__doc__,
        prep.OrdinalEncoder.__doc__,
        prep.FunctionTransformer.__doc__,
        prep.StandardScaler.__doc__,
        prep.RobustScaler.__doc__,
        prep.MinMaxScaler.__doc__,
        prep.PowerTransformer.__doc__,
        prep.PolynomialFeatures.__doc__,
        prep.SplineTransformer.__doc__,
        prep.QuantileTransformer.__doc__,
    ]
    text = [y for x in text for y in x.split("\n")]
    grid = {
        "kmeans__n_clusters": np.arange(3, 10),
    }
    pipe = make_pipeline(
        TfidfVectorizer(),
        decomposition.TruncatedSVD(random_state=864),
        cluster.KMeans(random_state=6),
    )
    search = ClusterOptimizer(pipe, grid, scoring="silhouette", error_score=-1)
    search.fit(text)
    check_is_fitted(
        search,
        [
            "results_",
            "best_score_",
            "best_index_",
            "best_estimator_",
            "best_params_",
            "labels_",
        ],
    )
    assert 0.5 < search.best_score_
    assert sorted(search.results_.keys()) == sorted(
        [
            "fit_time",
            "param_kmeans__n_clusters",
            "params",
            "rank_score",
            "score",
            "score_time",
            "noise_ratio",
        ]
    )
