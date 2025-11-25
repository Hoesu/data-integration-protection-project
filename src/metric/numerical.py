import numpy as np


def _l1_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    L1 거리(맨해튼 거리)를 계산합니다.

    Parameters
    ----------
    x : np.ndarray
        첫 번째 수치형 배열
    y : np.ndarray
        두 번째 수치형 배열

    Returns
    -------
    float
        두 배열 간의 L1 거리 (맨해튼 거리)

    Examples
    --------
    >>> _l1_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))
    9.0
    """
    # TODO: x와 y 배열의 shape이 같은지 검증
    # TODO: 각 요소별로 절댓값 차이 계산: |x_i - y_i|
    # TODO: 모든 절댓값 차이의 합 계산
    # TODO: 합계 반환
    pass


def _l2_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    L2 거리(유클리드 거리)를 계산합니다.

    Parameters
    ----------
    x : np.ndarray
        첫 번째 수치형 배열
    y : np.ndarray
        두 번째 수치형 배열

    Returns
    -------
    float
        두 배열 간의 L2 거리 (유클리드 거리)

    Examples
    --------
    >>> _l2_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))
    5.196...
    """
    # TODO: x와 y 배열의 shape이 같은지 검증
    # TODO: 각 요소별로 차이의 제곱 계산: (x_i - y_i)^2
    # TODO: 모든 제곱 차이의 합 계산
    # TODO: 합의 제곱근 계산 (sqrt)
    # TODO: 제곱근 값 반환
    pass


def _mahalanobis_distance(
    x: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray | None = None
) -> float:
    """
    Mahalanobis 거리를 계산합니다.

    Parameters
    ----------
    x : np.ndarray
        첫 번째 수치형 배열
    y : np.ndarray
        두 번째 수치형 배열
    cov : np.ndarray | None, optional
        공분산 행렬. None인 경우 단위 행렬로 간주

    Returns
    -------
    float
        두 배열 간의 Mahalanobis 거리

    Examples
    --------
    >>> _mahalanobis_distance(np.array([1, 2]), np.array([3, 4]), cov)
    2.828...
    """
    # TODO: x와 y 배열의 shape이 같은지 검증
    # TODO: cov가 None인 경우 단위 행렬로 초기화
    # TODO: cov 행렬의 역행렬 계산 (공분산 행렬의 역행렬)
    # TODO: 차이 벡터 계산: diff = x - y
    # TODO: Mahalanobis 거리 공식 적용: sqrt(diff^T * inv(cov) * diff)
    # TODO: 거리 값 반환
    pass


def compute_interrow_n_dist(
    x: dict,
    y: dict,
    properties: dict,
) -> float:
    """
    두 행의 수치형 컬럼들에 대한 거리를 계산합니다.

    Parameters
    ----------
    x : dict
        첫 번째 행의 수치형 컬럼 데이터 (컬럼명: 값)
    y : dict
        두 번째 행의 수치형 컬럼 데이터 (컬럼명: 값)
    properties : dict
        수치형 컬럼별 메트릭 정보를 담은 딕셔너리
        예: {'numeric': {'l2': ['col1', 'col2'], 'l1': ['col3'], 'mahalanobis': ['col4'], ...}}

    Returns
    -------
    float
        두 행 간의 수치형 거리 (가중 평균 또는 합)

    Examples
    --------
    >>> x = {'age': 25, 'income': 50000}
    >>> y = {'age': 30, 'income': 60000}
    >>> properties = {'numeric': {'l2': ['age'], 'l1': ['income']}}
    >>> compute_interrow_n_dist(x, y, properties)
    10005.0
    """
    # TODO: properties에서 'numeric' 키 확인
    # TODO: 각 메트릭별로 그룹화된 컬럼 리스트 추출
    # TODO: 각 메트릭 그룹에 대해:
    #   - 해당 메트릭에 속한 컬럼들의 값 배열 추출 (x, y에서)
    #   - 적절한 거리 함수 호출 (_l2_distance, _l1_distance, _mahalanobis_distance)
    #   - mahalanobis의 경우 공분산 행렬 정보 필요 (properties 또는 별도 계산)
    #   - 메트릭별 거리 값 저장
    # TODO: 모든 메트릭 그룹의 거리를 가중 평균 또는 합으로 계산
    # TODO: 최종 거리 값 반환
    pass