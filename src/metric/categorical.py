import logging
import numpy as np

logger = logging.getLogger('project.metric.categorical')


def _levenshtein_distance(
    x: np.ndarray[str],
    y: np.ndarray[str]
) -> float:
    """
    두 문자열 배열간의 Levenshtein 거리 평균을 계산합니다.

    Parameters
    ----------
    x : np.ndarray[str]
        첫 번째 문자열 배열
    y : np.ndarray[str]
        두 번째 문자열 배열

    Returns
    -------
    float
        두 배열 간의 Levenshtein 거리 평균값

    Examples
    --------
    >>> _levenshtein_distance(np.array(['abc', 'def']), np.array(['ab', 'de']))
    1.0
    """
    if len(x) != len(y):
        raise ValueError(f"x와 y의 길이가 일치해야 합니다. len(x)={len(x)}, len(y)={len(y)}")
    
    def levenshtein(s1: str, s2: str) -> int:
        """두 문자열 간의 Levenshtein 거리 계산 (동적 프로그래밍)"""
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distances = []
    for i in range(len(x)):
        s1 = str(x[i]) if x[i] is not None else ""
        s2 = str(y[i]) if y[i] is not None else ""
        dist = levenshtein(s1, s2)
        distances.append(dist)
    
    avg_distance = np.mean(distances) if distances else 0.0
    return float(avg_distance)


def _jaccard_distance(
    x: np.ndarray[str],
    y: np.ndarray[str]
) -> float:
    """
    두 문자열 배열간의 Jaccard 거리 평균을 계산합니다.

    Parameters
    ----------
    x : np.ndarray[str]
        첫 번째 문자열 배열
    y : np.ndarray[str]
        두 번째 문자열 배열

    Returns
    -------
    float
        두 배열 간의 Jaccard 거리 평균값 (0~1 범위)

    Examples
    --------
    >>> _jaccard_distance(np.array(['abc', 'def']), np.array(['ab', 'de']))
    0.5
    """
    if len(x) != len(y):
        raise ValueError(f"x와 y의 길이가 일치해야 합니다. len(x)={len(x)}, len(y)={len(y)}")
    
    def jaccard(s1: str, s2: str) -> float:
        """두 문자열 간의 Jaccard 거리 계산"""
        set1 = set(s1) if s1 else set()
        set2 = set(s2) if s2 else set()
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 1.0
        
        jaccard_coefficient = intersection / union
        jaccard_distance = 1.0 - jaccard_coefficient
        
        return jaccard_distance
    
    distances = []
    for i in range(len(x)):
        s1 = str(x[i]) if x[i] is not None else ""
        s2 = str(y[i]) if y[i] is not None else ""
        dist = jaccard(s1, s2)
        distances.append(dist)
    
    avg_distance = np.mean(distances) if distances else 0.0
    return float(avg_distance)


def _hamming_distance(
    x: np.ndarray[str],
    y: np.ndarray[str]
) -> float:
    """
    두 문자열 배열간의 Hamming 거리 평균을 계산합니다.

    Parameters
    ----------
    x : np.ndarray[str]
        첫 번째 문자열 배열
    y : np.ndarray[str]
        두 번째 문자열 배열

    Returns
    -------
    float
        두 배열 간의 Hamming 거리 평균값

    Examples
    --------
    >>> _hamming_distance(np.array(['abc', 'def']), np.array(['abd', 'deg']))
    1.0
    """
    if len(x) != len(y):
        raise ValueError(f"x와 y의 길이가 일치해야 합니다. len(x)={len(x)}, len(y)={len(y)}")
    
    def hamming(s1: str, s2: str) -> int:
        """두 문자열 간의 Hamming 거리 계산"""
        if len(s1) != len(s2):
            # 길이가 다르면 최대 길이로 패딩하고 다른 문자로 간주
            max_len = max(len(s1), len(s2))
            s1 = s1.ljust(max_len, '\0')
            s2 = s2.ljust(max_len, '\0')
        
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    distances = []
    for i in range(len(x)):
        s1 = str(x[i]) if x[i] is not None else ""
        s2 = str(y[i]) if y[i] is not None else ""
        dist = hamming(s1, s2)
        distances.append(dist)
    
    avg_distance = np.mean(distances) if distances else 0.0
    return float(avg_distance)

def compute_interrow_c_dist(
    x: dict,
    y: dict,
    properties: dict,
) -> float:
    """
    두 행의 범주형 컬럼들에 대한 거리를 계산합니다.

    Parameters
    ----------
    x : dict
        첫 번째 행의 범주형 컬럼 데이터 (컬럼명: 값)
    y : dict
        두 번째 행의 범주형 컬럼 데이터 (컬럼명: 값)
    properties : dict
        범주형 컬럼별 메트릭 정보를 담은 딕셔너리
        예: {'categorical': {'levenshtein': ['col1', 'col2'], 'jaccard': ['col3'], ...}}

    Returns
    -------
    float
        두 행 간의 범주형 거리 (가중 평균 또는 합)

    Examples
    --------
    >>> x = {'name': 'John', 'city': 'Seoul'}
    >>> y = {'name': 'Jane', 'city': 'Busan'}
    >>> properties = {'categorical': {'levenshtein': ['name'], 'jaccard': ['city']}}
    >>> compute_interrow_c_dist(x, y, properties)
    2.5
    """
    if 'categorical' not in properties:
        logger.debug('범주형 속성이 없어 거리 0.0 반환')
        return 0.0
    
    categorical_props = properties['categorical']
    if not categorical_props:
        logger.debug('범주형 속성이 비어있어 거리 0.0 반환')
        return 0.0
    
    logger.debug(f'범주형 거리 계산 시작: {len(categorical_props)}개 메트릭 그룹')
    distances = []
    
    for metric, columns in categorical_props.items():
        if not columns:
            continue
        
        # 해당 메트릭에 속한 컬럼들의 값 배열 추출
        x_values = np.array([x.get(col, "") for col in columns], dtype=object)
        y_values = np.array([y.get(col, "") for col in columns], dtype=object)
        
        if metric == 'levenshtein':
            dist = _levenshtein_distance(x_values, y_values)
            logger.debug(f'Levenshtein 거리 계산: {len(columns)}개 컬럼, 거리={dist:.4f}')
        elif metric == 'jaccard':
            dist = _jaccard_distance(x_values, y_values)
            logger.debug(f'Jaccard 거리 계산: {len(columns)}개 컬럼, 거리={dist:.4f}')
        elif metric == 'hamming':
            dist = _hamming_distance(x_values, y_values)
            logger.debug(f'Hamming 거리 계산: {len(columns)}개 컬럼, 거리={dist:.4f}')
        else:
            # 기본값으로 Levenshtein 거리 사용
            logger.warning(f'알 수 없는 메트릭 "{metric}", Levenshtein 거리 사용')
            dist = _levenshtein_distance(x_values, y_values)
        
        distances.append(dist)
    
    # 모든 메트릭 그룹의 거리를 합으로 계산
    total_distance = sum(distances) if distances else 0.0
    logger.debug(f'범주형 거리 계산 완료: 총 거리={total_distance:.4f}')
    
    return total_distance