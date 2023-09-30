"""Test matcher's modules."""

import torch

from matcher import fast_cosine_dist, knn


def test_fast_cosine_dist():
    """Test `fast_cosine_dist`."""

    query_series = torch.rand([100, 128])
    target_pool  = torch.rand([200, 128])

    distance = fast_cosine_dist(query_series, target_pool, torch.device("cpu"))

    assert distance.shape == (100, 200), "Wrong shape"


def test_knn():
    """Test `knn`."""

    query_series = torch.rand([100, 128])
    target_pool  = torch.rand([200, 128])

    knn_series = knn(query_series, target_pool, target_pool, 4, torch.device("cpu"))

    assert knn_series.shape == (100, 128), "Wrong shape"
