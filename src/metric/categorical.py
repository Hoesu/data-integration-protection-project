import numpy as np


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
    # TODO: x와 y 배열의 길이가 같은지 검증
    # TODO: 각 인덱스별로 문자열 쌍에 대해 Levenshtein 거리 계산
    # TODO: Levenshtein 거리 알고리즘 구현 (동적 프로그래밍 사용)
    # TODO: 모든 문자열 쌍의 거리 평균 계산
    # TODO: 평균값 반환
    pass


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
    # TODO: x와 y 배열의 길이가 같은지 검증
    # TODO: 각 인덱스별로 문자열 쌍에 대해 Jaccard 거리 계산
    # TODO: 각 문자열을 문자 집합으로 변환
    # TODO: 교집합과 합집합 계산하여 Jaccard 계수 도출
    # TODO: Jaccard 거리 = 1 - Jaccard 계수
    # TODO: 모든 문자열 쌍의 거리 평균 계산
    # TODO: 평균값 반환
    pass


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
    # TODO: x와 y 배열의 길이가 같은지 검증
    # TODO: 각 인덱스별로 문자열 쌍에 대해 Hamming 거리 계산
    # TODO: 문자열 길이가 같은지 검증 (Hamming 거리는 길이가 같아야 함)
    # TODO: 같은 위치의 문자가 다른 개수 세기
    # TODO: 모든 문자열 쌍의 거리 평균 계산
    # TODO: 평균값 반환
    pass

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
    # TODO: properties에서 'categorical' 키 확인
    # TODO: 각 메트릭별로 그룹화된 컬럼 리스트 추출
    # TODO: 각 메트릭 그룹에 대해:
    #   - 해당 메트릭에 속한 컬럼들의 값 배열 추출 (x, y에서)
    #   - 적절한 거리 함수 호출 (_levenshtein_distance, _jaccard_distance, _hamming_distance)
    #   - 메트릭별 거리 값 저장
    # TODO: 모든 메트릭 그룹의 거리를 가중 평균 또는 합으로 계산
    # TODO: 최종 거리 값 반환
    pass