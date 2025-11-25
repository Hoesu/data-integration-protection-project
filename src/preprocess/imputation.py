import pandas as pd


def _impute_row(
    target_index: int,
    neighbors: int,
    properties: dict,
    reference: pd.DataFrame,
) -> dict:
    """
    특정 행의 결측치를 가장 가까운 이웃 행들의 값을 기반으로 대체.

    Parameters
    ----------
    target_index : int
        결측치를 대체할 대상 행의 인덱스
    neighbors : int
        참조할 최근접 이웃 행의 개수
    properties : dict
        컬럼별 타입과 메트릭 정보를 담은 딕셔너리
    reference : pd.DataFrame
        거리 계산 및 대체값 추출에 사용할 참조 데이터프레임

    Returns
    -------
    dict
        결측치가 있는 컬럼명과 대체값의 쌍. 결측치가 없는 컬럼은 포함하지 않음.
        예: {'컬럼명1': 대체값1, '컬럼명2': 대체값2, ...}

    Examples
    --------
    >>> _impute_row(0, 5, properties, reference_df)
    {'age': 25.5, 'city': 'Seoul'}
    """
    # TODO: target_index 행에서 결측치가 존재하는 컬럼명 리스트 추출
    # TODO: reference 데이터프레임의 각 행과 target_index 행 간의 거리 계산
    #   - compute_interrow_dist 함수 호출하여 행간 거리 계산
    # TODO: 계산된 거리를 기준으로 가장 가까운 neighbors 개의 행 선택
    # TODO: 선택된 이웃 행들에서 각 결측 컬럼별로 대체값 계산:
    #   - 수치형 컬럼: 이웃 행들의 평균값 계산
    #   - 범주형 컬럼: 이웃 행들의 최빈값(mode) 계산
    # TODO: 컬럼명과 대체값의 쌍으로 딕셔너리 구성하여 반환
    pass

def impute_data(
    data: pd.DataFrame,
    properties: dict,
    config: dict
) -> pd.DataFrame:
    """
    데이터프레임의 모든 결측치를 K-Nearest Neighbors 기반으로 대체.

    Parameters
    ----------
    data : pd.DataFrame
        결측치를 대체할 데이터프레임
    properties : dict
        컬럼별 타입과 메트릭 정보를 담은 딕셔너리
    config : dict
        설정 딕셔너리 (neighbors 개수 등 포함)

    Returns
    -------
    pd.DataFrame
        결측치가 대체된 데이터프레임

    Examples
    --------
    >>> imputed_df = impute_data(df, properties, config)
    >>> imputed_df.isna().sum().sum()
    0
    """
    # TODO: 전체 데이터프레임에서 결측치가 하나도 없는 행들만 추출하여 reference 서브 데이터프레임 생성
    # TODO: reference가 비어있는 경우 예외 처리 또는 경고 메시지
    # TODO: 전체 데이터프레임에서 결측치가 하나 이상 존재하는 행의 인덱스 리스트 추출
    # TODO: config에서 neighbors 개수 추출 (기본값 설정 고려)
    # TODO: 결측치가 있는 각 행에 대해:
    #   - _impute_row 함수 호출하여 대체할 컬럼명과 대체값 쌍 획득
    #   - 획득한 대체값들을 원본 데이터프레임에 적용
    # TODO: 대체가 완료된 데이터프레임 반환
    pass
