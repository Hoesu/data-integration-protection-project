import logging

import pandas as pd

logger = logging.getLogger('project.preprocess.allocation')


def allocate_metadata(data: pd.DataFrame, config: dict) -> dict:
    """원천 데이터 컬럼 별로 수치형/범주형 분류와 최대값, 최소값 정보를 담은 메타데이터를 반환.

    수치형 컬럼만 최대, 최소값을 저장하고, 이는 추후 HEOM 계산에 사용됨

    Parameters
    ----------
    data : pd.DataFrame
        DB에서 조회한 원천 데이터프레임
    config : dict
        설정 딕셔너리

    Returns
    -------
    dict[str, dict[str, int | float | str]]
        컬럼별 메타데이터를 담은 딕셔너리

        각 컬럼의 메타데이터 딕셔너리 구조:
        - 'type': str
            컬럼의 데이터 타입. 'numeric', 'categorical', 'exclude' 중 하나.
        - 'max': int | float
            수치형 컬럼의 경우 최대값. 범주형 컬럼의 경우 포함되지 않음.
        - 'min': int | float
            수치형 컬럼의 경우 최소값. 범주형 컬럼의 경우 포함되지 않음.

        범주형 컬럼은 'max'와 'min' 키가 포함되지 않으며, 수치형 컬럼만 이 값들이 포함됨.
        exclude 컬럼은 후속 연산에서 제외하기 위해 표시하는 용도.

    Examples
    --------
    >>> metadata = allocate_metadata(data, config)
    >>> metadata['age']
    {'type': 'numeric', 'max': 80.0, 'min': 18.0}
    >>> metadata['gender']
    {'type': 'categorical'}
    """
    logger.debug('Starting metadata allocation')
    exclude_columns = config['preprocess']['exclude_columns']
    metadata = {}
    for col in data.columns:
        if col in exclude_columns:
            metadata[col] = {'type': 'exclude'}
        elif pd.api.types.is_numeric_dtype(data[col]):
            metadata[col] = {'type': 'numeric', 'max': data[col].max(), 'min': data[col].min()}
        else:
            metadata[col] = {'type': 'categorical'}

    for item in metadata.items():
        logger.debug(f'Column: {item[0]}, Type: {item[1]["type"]}')
    logger.debug('Metadata allocation complete')
    return metadata