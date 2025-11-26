import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.metric import compute_interrow_dist

logger = logging.getLogger('project.preprocess.imputation')


def _impute_row(
    target_row: pd.Series,
    num_neighbors: int,
    properties: dict,
    reference: pd.DataFrame,
) -> dict:
    """특정 행의 결측치를 K-NN 기반으로 대체.

    Parameters
    ----------
    target_row : pd.Series
        결측치를 대체할 대상 행
    num_neighbors : int
        참조할 최근접 이웃 행의 개수
    properties : dict
        컬럼별 타입과 메트릭 정보 딕셔너리
    reference : pd.DataFrame
        거리 계산 및 대체값 추출에 사용할 참조 데이터프레임

    Returns
    -------
    dict[str, Any]
        결측치가 있는 컬럼명과 대체값의 쌍. 결측치가 없는 컬럼은 포함하지 않음.
    """
    missing_columns = target_row[target_row.isna()].index.tolist()

    target_dict = {
        k: v for k, v in target_row.to_dict().items()
        if k not in missing_columns
    }

    reference_values = reference[missing_columns]
    reference = reference.drop(columns=missing_columns)

    distances = []
    for idx in reference.index:
        ref_dict = reference.loc[idx].to_dict()
        dist = compute_interrow_dist(target_dict, ref_dict, properties)
        distances.append((idx, dist))

    distances.sort(key=lambda x: x[1])
    neighbor_indices = [idx for idx, _ in distances[:num_neighbors]]

    # 컬럼 타입을 미리 매핑
    column_types = {}
    if 'numeric' in properties:
        for cols in properties['numeric'].values():
            for col in cols:
                if col in missing_columns:
                    column_types[col] = 'numeric'
    
    for col in missing_columns:
        if col not in column_types:
            column_types[col] = 'categorical'

    # 이웃 값들을 한 번에 추출
    neighbor_data = reference_values.loc[neighbor_indices]

    # 각 컬럼별로 대체값 계산
    imputed_values = {}
    for col in missing_columns:
        neighbor_values = neighbor_data[col].dropna().tolist()

        if column_types[col] == 'numeric':
            imputed_values[col] = np.mean(neighbor_values)
        else:
            counter = Counter(neighbor_values)
            imputed_values[col] = counter.most_common(1)[0][0]

    logger.debug(f'결측치 대체 완료: {len(imputed_values)}개 컬럼 대체됨')
    return imputed_values

def impute_data(
    data: pd.DataFrame,
    properties: dict,
    config: dict
) -> pd.DataFrame:
    """데이터프레임의 모든 결측치를 K-NN 기반으로 대체.

    Parameters
    ----------
    data : pd.DataFrame
        결측치를 대체할 데이터프레임
    properties : dict
        컬럼별 타입과 메트릭 정보 딕셔너리
    config : dict
        전처리 설정 딕셔너리 (imputation_neighbors 등 포함)

    Returns
    -------
    pd.DataFrame
        결측치가 대체된 데이터프레임
    """
    logger.info('결측치 대체 프로세스 시작')
    reference = data[data.notna().all(axis=1)].copy()
    na_indices = data[data.isna().any(axis=1)].index.tolist()
    num_neighbors = config['preprocess']['imputation_neighbors']

    if len(reference) == 0:
        error_msg = (
            '결측치가 없는 참조 행이 없습니다. '
            '결측치 대체를 수행할 수 없습니다.'
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    elif len(reference) == len(data):
        logger.info('결측치가 없어 대체 불필요')
        return data

    else:
        logger.info(f'참조 행 개수: {len(reference)}개')
        logger.info(f'결측치가 있는 행 개수: {len(na_indices)}개')
        logger.info(f'이웃 개수: {num_neighbors}개')

    for na_index in na_indices:
        target_row = data.loc[na_index]
        imputed_values = _impute_row(
            target_row, num_neighbors, properties, reference
        )
        for col, value in imputed_values.items():
            data.at[na_index, col] = value

    return data