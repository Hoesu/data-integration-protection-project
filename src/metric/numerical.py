import logging
import numpy as np

logger = logging.getLogger('project.metric.numerical')


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
    if x.shape != y.shape:
        raise ValueError(f"x와 y의 shape이 일치해야 합니다. x.shape={x.shape}, y.shape={y.shape}")
    
    diff = np.abs(x - y)
    distance = np.sum(diff)
    
    return float(distance)


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
    if x.shape != y.shape:
        raise ValueError(f"x와 y의 shape이 일치해야 합니다. x.shape={x.shape}, y.shape={y.shape}")
    
    diff_squared = (x - y) ** 2
    sum_squared = np.sum(diff_squared)
    distance = np.sqrt(sum_squared)
    
    return float(distance)


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
    if x.shape != y.shape:
        raise ValueError(f"x와 y의 shape이 일치해야 합니다. x.shape={x.shape}, y.shape={y.shape}")
    
    if cov is None:
        cov = np.eye(len(x))
    
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # 역행렬이 존재하지 않는 경우 의사역행렬 사용
        inv_cov = np.linalg.pinv(cov)
    
    diff = x - y
    distance = np.sqrt(diff.T @ inv_cov @ diff)
    
    return float(distance)


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
    if 'numeric' not in properties:
        logger.debug('수치형 속성이 없어 거리 0.0 반환')
        return 0.0
    
    numeric_props = properties['numeric']
    if not numeric_props:
        logger.debug('수치형 속성이 비어있어 거리 0.0 반환')
        return 0.0
    
    logger.debug(f'수치형 거리 계산 시작: {len(numeric_props)}개 메트릭 그룹')
    distances = []
    
    for metric, columns in numeric_props.items():
        if not columns:
            continue
        
        # 해당 메트릭에 속한 컬럼들의 값 배열 추출
        x_values = np.array([x.get(col, 0) for col in columns])
        y_values = np.array([y.get(col, 0) for col in columns])
        
        if metric == 'l1':
            dist = _l1_distance(x_values, y_values)
            logger.debug(f'L1 거리 계산: {len(columns)}개 컬럼, 거리={dist:.4f}')
        elif metric == 'l2':
            dist = _l2_distance(x_values, y_values)
            logger.debug(f'L2 거리 계산: {len(columns)}개 컬럼, 거리={dist:.4f}')
        elif metric == 'mahalanobis':
            # Mahalanobis의 경우 공분산 행렬이 필요하지만, 
            # properties에 없으면 단위 행렬 사용
            cov = None
            if 'covariance' in properties.get('numeric', {}):
                cov = properties['numeric']['covariance'].get(metric, None)
            dist = _mahalanobis_distance(x_values, y_values, cov)
            logger.debug(f'Mahalanobis 거리 계산: {len(columns)}개 컬럼, 거리={dist:.4f}')
        else:
            # 기본값으로 L2 거리 사용
            logger.warning(f'알 수 없는 메트릭 "{metric}", L2 거리 사용')
            dist = _l2_distance(x_values, y_values)
        
        distances.append(dist)
    
    # 모든 메트릭 그룹의 거리를 합으로 계산
    total_distance = sum(distances) if distances else 0.0
    logger.debug(f'수치형 거리 계산 완료: 총 거리={total_distance:.4f}')
    
    return total_distance