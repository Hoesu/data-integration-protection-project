import logging

import pandas as pd

logger = logging.getLogger('project.preprocess.allocation')

def _allocate_type(data: pd.DataFrame, config: dict) -> list[str]:
    """각 컬럼의 데이터 타입을 판단하여 리스트로 반환.

    Parameters
    ----------
    data : pd.DataFrame
        타입을 판단할 데이터프레임
    config : dict
        전처리 설정 딕셔너리

    Returns
    -------
    list[str | None]
        각 컬럼의 타입 리스트. 'numeric', 'categorical', 또는 None 반환.
    """
    logger.debug(f'컬럼 타입 할당 시작: {len(data.columns)}개 컬럼')

    exclude_columns = config['preprocess']['exclude_columns']
    if exclude_columns:
        logger.debug(f'제외할 컬럼: {exclude_columns}')

    types = []
    numeric_count = 0
    categorical_count = 0
    excluded_count = 0

    for col in data.columns:
        if col in exclude_columns:
            types.append(None)
            excluded_count += 1
            continue

        dtype = data[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            types.append('numeric')
            numeric_count += 1
        else:
            types.append('categorical')
            categorical_count += 1

    logger.info(
        f'컬럼 타입 할당 완료: '
        f'수치형={numeric_count}, '
        f'범주형={categorical_count}, '
        f'제외={excluded_count}'
    )
    return types

def _allocate_metric(types: list[str], config: dict) -> list[str]:
    """각 컬럼 타입에 맞는 거리 메트릭을 할당하여 리스트로 반환.

    Parameters
    ----------
    types : list[str]
        각 컬럼의 타입 리스트 ('numeric', 'categorical', 또는 None)
    config : dict
        전처리 설정 딕셔너리

    Returns
    -------
    list[str | None]
        각 컬럼에 할당된 거리 메트릭 리스트.
        수치형 컬럼: 'l2', 'l1', 'mahalanobis' 중 하나
        범주형 컬럼: 'levenshtein', 'jaccard', 'hamming' 중 하나
        제외된 컬럼: None
    """
    numeric_metric = config['preprocess']['numeric_metric']
    categorical_metric = config['preprocess']['categorical_metric']

    logger.debug(
        f'메트릭 할당: '
        f'수치형={numeric_metric}, '
        f'범주형={categorical_metric}'
    )

    metrics = []

    for col_type in types:
        if col_type is None:
            metrics.append(None)
        elif col_type == 'numeric':
            metrics.append(numeric_metric)
        elif col_type == 'categorical':
            metrics.append(categorical_metric)
        else:
            metrics.append(None)
    return metrics

def allocate_properties(data: pd.DataFrame, config: dict) -> dict:
    """각 컬럼의 타입과 메트릭을 할당하여 속성 딕셔너리를 반환.

    Parameters
    ----------
    data : pd.DataFrame
        속성을 할당할 데이터프레임
    config : dict
        전처리 설정 딕셔너리

    Returns
    -------
    dict[str, dict[str, list[str]]]
        컬럼 타입별, 메트릭별로 그룹화된 컬럼명 딕셔너리.
        구조: {'categorical': {메트릭명: [컬럼명, ...]},
               'numeric': {메트릭명: [컬럼명, ...]}}
    """
    logger.info(f'속성 할당 시작: {len(data.columns)}개 컬럼')
    types = _allocate_type(data, config)
    metrics = _allocate_metric(types, config)
    result = {
        'categorical': {},
        'numeric': {}
    }

    added_count = 0
    for col, col_type, metric in zip(data.columns, types, metrics):
        if col_type is None or metric is None:
            continue

        if col_type == 'categorical':
            if metric not in result['categorical']:
                result['categorical'][metric] = []
            result['categorical'][metric].append(col)
            added_count += 1
        elif col_type == 'numeric':
            if metric not in result['numeric']:
                result['numeric'][metric] = []
            result['numeric'][metric].append(col)
            added_count += 1

    logger.info(f'속성 할당 완료: 총 {added_count}개 컬럼 할당됨')

    numeric_metrics = list(result['numeric'].keys())
    categorical_metrics = list(result['categorical'].keys())
    logger.debug(
        f'속성 구조: '
        f'수치형 메트릭={numeric_metrics}, '
        f'범주형 메트릭={categorical_metrics}'
    )
    return result