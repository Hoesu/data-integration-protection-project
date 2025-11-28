import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.metric import heom_distance

logger = logging.getLogger('project.preprocess.imputation')


def _impute_row(
    target_row: dict,
    reference: pd.DataFrame,
    imputation_neighbors: int,
    distance_threshold: float,
    metadata: dict
) -> dict:
    """단일 행의 결측치를 K-NN 기반으로 대체.

    HEOM 거리를 사용하여 참조 데이터에서 가장 가까운 이웃을 찾고,
    해당 이웃들의 값을 기반으로 결측치를 대체합니다.
    - 수치형 변수: 이웃들의 평균값 사용
    - 범주형 변수: 이웃들의 최빈값 사용

    Parameters
    ----------
    target_row : dict
        결측치를 대체할 대상 행 (컬럼명: 값)
    reference : pd.DataFrame
        참조 데이터프레임 (결측치가 없는 행들)
    imputation_neighbors : int
        사용할 최대 이웃 개수 (K)
    distance_threshold : float
        허용할 최대 거리 임계값
    metadata : dict
        컬럼별 메타데이터 딕셔너리
        {column_name: {'type': 'numeric' | 'categorical' | 'exclude', ...}}

    Returns
    -------
    dict
        대체된 값들의 딕셔너리 {column_name: imputed_value}
        조건을 만족하는 이웃이 없으면 빈 딕셔너리 반환

    Notes
    -----
    - 참조 데이터는 거리 기준으로 정렬된 후, 임계값 이하의 이웃만 선택됩니다.
    - 선택된 이웃 중 최대 K개만 사용됩니다.
    - reference는 내부에서 수정되므로 copy()로 전달해야 합니다.
    """
    imputed_values = {}
    ## target_row에서 결측치가 있는 컬럼명 찾기
    missing_columns = [col for col in target_row if target_row[col] is None]
    ## 레퍼런스 테이블 모든 행에 대하여 target_row와의 heom distance 계산
    distances = []
    for i in range(len(reference)):
        reference_dict = reference.iloc[i].to_dict()
        distance = heom_distance(
            x=target_row,
            y=reference_dict,
            metadata=metadata
        )
        distances.append(distance)
    ## 거리 컬럼을 한 번에 추가
    reference['distance'] = distances
    ## 레퍼런스 테이블을 distance 기준으로 정렬하기
    reference = reference.sort_values(by='distance')
    ## distance threshold 벗어나는 행 제거하기
    reference = reference[reference['distance'] <= distance_threshold]
    ## 레퍼런스 테이블에서 [0:num_neighbors] 범위만 선택하기
    reference = reference.iloc[0:imputation_neighbors]
    ## 사용할 수 있는 참조 행이 남아있지 않으면 빈값 반환
    if len(reference) == 0:
        logger.debug('조건을 만족하는 참조 행이 없습니다.')
        return imputed_values
    ## 각 결측 컬럼에 대하여 대체값 계산
    for column in missing_columns:
        if metadata[column]['type'] == 'numeric':
            ## 수치형 컬럼인 경우 평균값 계산
            average = reference[column].mean()
            imputed_values[column] = average
        elif metadata[column]['type'] == 'categorical':
            ## 범주형 컬럼인 경우 최빈값 계산
            counter = Counter(reference[column])
            imputed_values[column] = counter.most_common(1)[0][0]
    return imputed_values


def impute_data(
    data: pd.DataFrame,
    metadata: dict,
    config: dict
) -> pd.DataFrame:
    """데이터프레임의 모든 결측치를 HEOM 거리 기반 K-NN 알고리즘으로 대체.

    전체 데이터에서 결측치가 없는 행을 참조 데이터로 사용하여,
    결측치가 있는 각 행에 대해 가장 가까운 이웃을 찾아 값을 대체합니다.

    Parameters
    ----------
    data : pd.DataFrame
        결측치를 대체할 데이터프레임
    metadata : dict
        컬럼별 메타데이터 딕셔너리
        {column_name: {'type': 'numeric' | 'categorical' | 'exclude', ...}}
    config : dict
        설정 딕셔너리. 다음 키를 포함해야 합니다:
        - 'preprocess': dict
            - 'imputation_neighbors': int
                사용할 최대 이웃 개수 (K)
            - 'threshold_percentage': float
                최대 거리의 몇 퍼센트까지 허용할지 (0.0 ~ 1.0)

    Returns
    -------
    pd.DataFrame
        결측치가 대체된 데이터프레임. 원본 데이터프레임이 수정됩니다.

    Notes
    -----
    - 참조 데이터가 없거나 결측치가 없으면 원본 데이터를 그대로 반환합니다.
    - 각 행의 결측치는 독립적으로 처리됩니다.
    - 거리 임계값은 최대 가능한 거리(maximum_distance)에 threshold_percentage를 곱한 값입니다.
    - maximum_distance = sqrt(제외되지 않은 컬럼 개수)
    """
    ## 참조 데이터로 사용할 결측치가 없는 행만 추출하여 만든 데이터프레임
    reference = data[data.notna().all(axis=1)].copy()
    ## 결측치가 있는 행의 인덱스 리스트
    na_indices = data[data.isna().any(axis=1)].index.tolist()
    ## 하나의 행에 대하여 대조군으로 사용할 최대 이웃 개수
    imputation_neighbors = min(len(reference), config['preprocess']['imputation_neighbors'])
    ## 최대 가능한 거리 차이의 몇 퍼센트까지 허용할지 결정하는 값
    threshold_percentage = config['preprocess']['threshold_percentage']
    ## 현재 데이터 기준으로 최대 가능한 거리 차이 (예: 5개 컬럼이면 최대 거리 차이는 2.236)
    maximum_distance = np.float32(np.sqrt(len([col for col in metadata if metadata[col]['type'] != 'exclude'])))
    ## 데이터 보간을 위해 이웃을 찾을 때 허용하는 거리 차이
    threshold_distance = np.float32(maximum_distance * threshold_percentage)

    if len(reference) == 0:
        logger.warning('결측치를 대체할 참조 데이터가 없습니다.')
        return data
    elif len(reference) == len(data):
        logger.info('결측치가 없어 보간 작업이 불필요합니다.')
        return data
    else:
        logger.info('보간 작업을 시작합니다.')
        logger.info(f'참조 행 개수: {len(reference)}개')
        logger.info(f'결측치가 있는 행 개수: {len(na_indices)}개')
        logger.info(f'이웃 개수: {imputation_neighbors}개')
        logger.info(f'최대 거리 차이: {maximum_distance}')
        logger.info(f'거리 차이 임계값 비율: {threshold_percentage}')
        logger.info(f'거리 차이 임계값: {threshold_distance}')

        for na_index in na_indices:
            target_row = data.iloc[na_index].to_dict()
            imputed_values = _impute_row(
                target_row=target_row,
                reference=reference.copy(),
                imputation_neighbors=imputation_neighbors,
                distance_threshold=threshold_distance,
                metadata=metadata
            )
            for col, value in imputed_values.items():
                data.at[na_index, col] = value
        return data
