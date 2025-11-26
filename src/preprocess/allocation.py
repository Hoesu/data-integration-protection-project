import pandas as pd


def _allocate_type(data: pd.DataFrame, config: dict) -> list[str]:
    """
    정렬된 data 컬럼별로 수치형인지, 범주형인지 판단하여 수치형/범주형 여부를 반환.

    Parameters
    ----------
    data : pd.DataFrame
        판단할 데이터프레임
    config : dict
        설정 딕셔너리

    Returns
    -------
    list[str]
        각 컬럼의 타입 리스트. 'numeric' 또는 'categorical' 반환.
        연산에서 제외시킬 컬럼은 None 반환.

    Examples
    --------
    >>> _allocate_type(df)
    ['numeric', 'categorical', 'numeric', None, ...]
    """
    # TODO: 데이터프레임 순회하며 각 컬럼의 타입을 기반으로 수치형/범주형 구분
    # TODO: 설정 딕셔너리에 따라 연산에서 제외할 컬럼(예: ID 컬럼 등)은 None 반환
    # TODO: 판단 결과를 리스트로 반환 (컬럼 순서 유지)
    pass

def _allocate_metric(types: list[str], config: dict) -> list[str]:
    """
    각 컬럼별로 거리 메트릭 지정하여 거리 메트릭의 종류를 리스트로 반환.

    Parameters
    ----------
    types : list[str]
        각 컬럼의 타입 리스트
    config : dict
        설정 딕셔너리

    Returns
    -------
    list[str]
        각 컬럼에 할당된 거리 메트릭 리스트.
        수치형: 'l2', 'l1', 'mahalanobis' 중 선택
        범주형: 'levenshtein', 'jaccard', 'hamming' 중 선택
        연산에서 제외시킬 컬럼은 None 반환.

    Examples
    --------
    >>> _allocate_metric(types, config)
    ['l2', 'jaccard', 'l2', None, 'jaccard', ...]
    """
    # TODO: types 리스트를 순회하며 각 컬럼의 타입에 따라 거리 메트릭 지정.
    # TODO: 설정 딕셔너리에 따라 수치형 컬럼의 거리 메트릭 지정.
    # TODO: 설정 딕셔너리에 따라 범주형 컬럼의 거리 메트릭 지정.
    # TODO: 수치형/범주형 구분이 None이라면 None 반환.
    # TODO: 할당된 메트릭을 리스트로 반환 (컬럼 순서 유지)
    pass

def allocate_properties(data: pd.DataFrame, config: dict) -> dict:
    """
    데이터프레임의 각 컬럼에 대해 타입과 메트릭을 할당하여 속성 딕셔너리를 반환.

    Parameters
    ----------
    data : pd.DataFrame
        속성을 할당할 데이터프레임
    config : dict
        설정 딕셔너리

    Returns
    -------
    dict
        {
            'categorical': {
                'levenshtein': ['컬럼명1', '컬럼명2', ...],
                'jaccard': ['컬럼명3', '컬럼명4', ...],
                'hamming': [...],
                ...
            },
            'numeric': {
                'l2': ['컬럼명5', '컬럼명6', ...],
                'l1': ['컬럼명7', '컬럼명8', ...],
                'mahalanobis': ['컬럼명9', '컬럼명10', ...],
                ...
            }
        }
    """
    # TODO: data 컬럼 첫행 뽑아서 컬럼명 순으로 정렬
    # TODO: _allocate_type 함수 호출하여 각 컬럼의 타입 리스트 획득
    # TODO: _allocate_metric 함수 호출하여 각 컬럼의 메트릭 리스트 획득
    # TODO: 결과 딕셔너리 초기화: {'categorical': {}, 'numeric': {}}
    # TODO: 각 컬럼을 순회하며:
    #   - 타입이 None이거나 메트릭이 None인 경우 스킵
    #   - 타입에 따라 'categorical' 또는 'numeric' 그룹에 추가
    #   - 메트릭을 키로 사용하여 해당 메트릭의 컬럼 리스트에 추가
    # TODO: 최종 딕셔너리 반환
    pass