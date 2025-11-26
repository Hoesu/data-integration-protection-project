import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('project.preprocess.normalization')


def normalize_data(
    data: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    데이터프레임의 수치형 컬럼들을 Min-Max 정규화합니다.

    범주형 컬럼과 제외된 컬럼은 정규화하지 않고 그대로 유지합니다.
    정규화 공식: (x - min) / (max - min)

    Parameters
    ----------
    data : pd.DataFrame
        정규화할 데이터프레임
    config : dict
        전처리 설정 딕셔너리 (exclude_columns 등 포함)

    Returns
    -------
    pd.DataFrame
        수치형 컬럼이 정규화된 데이터프레임

    Examples
    --------
    >>> data = pd.DataFrame({'age': [20, 30, 40], 'name': ['A', 'B', 'C']})
    >>> config = {'preprocess': {'exclude_columns': []}}
    >>> normalized = normalize_data(data, config)
    >>> normalized['age'].min()
    0.0
    >>> normalized['age'].max()
    1.0
    """
    logger.info('데이터 정규화 프로세스 시작')
    
    # 데이터프레임 복사 (원본 보존)
    normalized_data = data.copy()
    
    # 제외할 컬럼 목록 가져오기
    exclude_columns = config.get('preprocess', {}).get('exclude_columns', [])
    if exclude_columns:
        logger.debug(f'제외할 컬럼: {exclude_columns}')
    
    # 정규화할 컬럼 개수 추적
    normalized_count = 0
    skipped_count = 0
    
    # 각 컬럼에 대해 정규화 수행
    for col in data.columns:
        # 제외된 컬럼은 스킵
        if col in exclude_columns:
            logger.debug(f'컬럼 "{col}" 제외됨 (exclude_columns)')
            skipped_count += 1
            continue
        
        # 수치형 컬럼만 정규화
        if not pd.api.types.is_numeric_dtype(data[col]):
            logger.debug(f'컬럼 "{col}" 스킵됨 (범주형)')
            skipped_count += 1
            continue
        
        # 결측치가 있는 컬럼은 스킵 (이미 imputation이 완료되었지만 안전장치)
        if data[col].isna().any():
            logger.warning(f'컬럼 "{col}"에 결측치가 있어 정규화 스킵')
            skipped_count += 1
            continue
        
        # Min-Max 정규화 수행
        col_min = data[col].min()
        col_max = data[col].max()
        col_range = col_max - col_min
        
        # 상수 컬럼 (max == min)인 경우 0으로 설정
        if col_range == 0:
            logger.debug(f'컬럼 "{col}"는 상수값이므로 0으로 설정')
            normalized_data[col] = 0.0
        else:
            # Min-Max 정규화: (x - min) / (max - min)
            normalized_data[col] = (data[col] - col_min) / col_range
            logger.debug(
                f'컬럼 "{col}" 정규화 완료: '
                f'원본 범위=[{col_min:.4f}, {col_max:.4f}], '
                f'정규화 범위=[0.0, 1.0]'
            )
        
        normalized_count += 1
    
    logger.info(
        f'데이터 정규화 완료: '
        f'정규화된 컬럼={normalized_count}개, '
        f'스킵된 컬럼={skipped_count}개'
    )
    
    return normalized_data