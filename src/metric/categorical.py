import numpy as np


def _levenshtein_distance(
    x: np.ndarray[str],
    y: np.ndarray[str]
) -> float:
    """
    두 문자열 배열간의 Levenshtein 거리 평균을 계산합니다.
    """
    pass


def _jaccard_distance(
    x: np.ndarray[str],
    y: np.ndarray[str]
) -> float:
    """
    두 문자열 배열간의 Jaccard 거리 평균을 계산합니다.
    """
    pass


def _hamming_distance(
    x: np.ndarray[str],
    y: np.ndarray[str]
) -> float:
    """
    두 문자열 배열간의 Hamming 거리 평균을 계산합니다.
    """
    pass

def compute_interrow_c_dist(
    x: dict,
    y: dict,
    properties: dict,
) -> float:
    """
    두 범주형 자료 배열간의 거리를 계산합니다.
    """
    pass