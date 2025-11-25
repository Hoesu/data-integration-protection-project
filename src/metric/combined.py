import numpy as np
import pandas as pd


def compute_interrow_dist(
    x: dict,
    y: dict,
    properties: dict,
) -> float:
    """
    두 행 간의 전체 거리를 계산합니다 (수치형 + 범주형).

    Parameters
    ----------
    x : dict
        첫 번째 행의 데이터 (컬럼명: 값)
    y : dict
        두 번째 행의 데이터 (컬럼명: 값)
    properties : dict
        컬럼별 타입과 메트릭 정보를 담은 딕셔너리
        예: {
            'numeric': {'l2': [...], 'l1': [...], ...},
            'categorical': {'levenshtein': [...], 'jaccard': [...], ...}
        }

    Returns
    -------
    float
        두 행 간의 전체 거리 (수치형 거리 + 범주형 거리)

    Examples
    --------
    >>> x = {'age': 25, 'name': 'John'}
    >>> y = {'age': 30, 'name': 'Jane'}
    >>> properties = {'numeric': {'l2': ['age']}, 'categorical': {'levenshtein': ['name']}}
    >>> compute_interrow_dist(x, y, properties)
    5.5
    """
    # TODO: compute_interrow_n_dist 함수 호출하여 수치형 거리 계산
    # TODO: compute_interrow_c_dist 함수 호출하여 범주형 거리 계산
    # TODO: 수치형과 범주형 거리를 가중 평균 또는 합으로 결합
    # TODO: 최종 거리 값 반환
    pass

def compute_pairwise_dist(
    data: pd.DataFrame,
    properties: dict,
) -> np.ndarray[float]:
    """
    전체 데이터프레임에서 모든 행 쌍에 대한 거리 행렬을 계산합니다.

    Parameters
    ----------
    data : pd.DataFrame
        거리를 계산할 데이터프레임
    properties : dict
        컬럼별 타입과 메트릭 정보를 담은 딕셔너리

    Returns
    -------
    np.ndarray[float]
        모든 행 쌍에 대한 거리 행렬 (n x n 형태, n은 행 개수)
        dist_matrix[i, j]는 i번째 행과 j번째 행 간의 거리

    Examples
    --------
    >>> dist_matrix = compute_pairwise_dist(df, properties)
    >>> dist_matrix.shape
    (100, 100)
    """
    # TODO: 데이터프레임의 행 개수 확인
    # TODO: n x n 크기의 거리 행렬 초기화 (numpy 배열)
    # TODO: 모든 행 쌍 (i, j)에 대해:
    #   - i번째 행과 j번째 행을 딕셔너리 형태로 변환
    #   - compute_interrow_dist 함수 호출하여 거리 계산
    #   - 거리 행렬의 [i, j] 위치에 거리 값 저장
    #   - 대칭 행렬이므로 [j, i]에도 동일한 값 저장 (최적화 고려)
    # TODO: 대각선 요소는 0으로 설정 (자기 자신과의 거리)
    # TODO: 거리 행렬 반환
    pass