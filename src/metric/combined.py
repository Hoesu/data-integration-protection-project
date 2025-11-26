import numpy as np
import pandas as pd

from src.metric.numerical import compute_interrow_n_dist
from src.metric.categorical import compute_interrow_c_dist


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
    # 수치형 거리 계산
    numeric_dist = compute_interrow_n_dist(x, y, properties)
    
    # 범주형 거리 계산
    categorical_dist = compute_interrow_c_dist(x, y, properties)
    
    # 수치형과 범주형 거리를 합으로 결합
    total_dist = numeric_dist + categorical_dist
    
    return total_dist

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
    # 데이터프레임의 행 개수 확인
    n = len(data)
    
    # n x n 크기의 거리 행렬 초기화
    dist_matrix = np.zeros((n, n), dtype=float)
    
    # 모든 행 쌍 (i, j)에 대해 거리 계산
    # 대칭 행렬이므로 i <= j인 경우만 계산하고 최적화
    for i in range(n):
        # 자기 자신과의 거리는 0 (대각선 요소)
        dist_matrix[i, i] = 0.0
        
        # i < j인 경우만 계산하고 대칭 위치에도 동일한 값 저장
        for j in range(i + 1, n):
            # i번째 행과 j번째 행을 딕셔너리 형태로 변환
            x_dict = data.iloc[i].to_dict()
            y_dict = data.iloc[j].to_dict()
            
            # compute_interrow_dist 함수 호출하여 거리 계산
            distance = compute_interrow_dist(x_dict, y_dict, properties)
            
            # 거리 행렬의 [i, j]와 [j, i] 위치에 거리 값 저장 (대칭 행렬)
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    
    return dist_matrix