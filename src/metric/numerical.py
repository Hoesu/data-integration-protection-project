import numpy as np


def _l1_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    L1 거리(맨해튼 거리)를 계산합니다.
    """
    pass


def _l2_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    L2 거리(유클리드 거리)를 계산합니다.
    """
    pass


def _mahalanobis_distance(
    x: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray | None = None
) -> float:
    """
    Mahalanobis 거리를 계산합니다.
    """
    pass


def compute_interrow_n_dist(
    x: dict,
    y: dict,
    properties: dict,
) -> float:
    """
    두 수치형 자료 배열간의 거리를 계산합니다.
    """
    pass