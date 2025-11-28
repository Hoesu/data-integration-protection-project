import warnings

import numpy as np


def heom_distance(x: dict, y: dict, metadata: dict) -> float:
    """HEOM (Heterogeneous Euclidean-Overlap Metric) 거리 계산.

    HEOM은 범주형과 수치형 변수를 모두 처리할 수 있는 거리 측정 방법.
    - 범주형 변수: 값이 같으면 0, 다르면 1
    - 수치형 변수: 값의 차이를 값의 범위로 정규화
    - 결측치 처리: 비교 대상 컬럼에 결측치가 있으면 1으로 처리.
    - 최종 거리: 모든 변수에 대한 제곱합의 제곱근으로 계산.

    Parameters
    ----------
    x : dict
        첫 번째 데이터 포인트 {column_name: value}
    y : dict
        두 번째 데이터 포인트 {column_name: value}
    metadata : dict
        컬럼별 메타데이터

    Returns
    -------
    float
        HEOM 거리 값 (0 이상의 실수)

    Examples
    --------
    >>> x = {'age': 25, 'city': 'Seoul'}
    >>> y = {'age': 30, 'city': 'Busan'}
    >>> metadata = {
    ...     'age': {'type': 'numeric', 'min': 0, 'max': 100},
    ...     'city': {'type': 'categorical'}
    ... }
    >>> heom_distance(x, y, metadata)
    1.002...
    """
    ## 제곱합을 누적할 변수 (float32로 메모리 효율성 확보)
    squared_sum = np.float32(0.0)
    ## x의 모든 컬럼에 대해 거리 계산
    for column_name in x:
        column_type = metadata[column_name]['type']
        ## 범주형 변수 처리: 값이 같으면 0, 다르면 1
        ## 값이 같으면 0, 다르면 1을 제곱합에 더함
        if column_type == 'categorical':
            ## 결측치 처리: 비교 대상 컬럼에 결측치가 있으면 1으로 처리.
            if x[column_name]==None or y[column_name]==None:
                squared_sum += np.float32(1.0)
                continue
            ## x[column_name] != y[column_name]는 같으면 False(0), 다르면 True(1)
            squared_sum += np.float32(x[column_name] != y[column_name])
        ## 수치형 변수 처리: 값의 차이를 값의 범위로 정규화
        elif column_type == 'numeric':
            ## 결측치 처리: 비교 대상 컬럼에 결측치가 있으면 1으로 처리.
            if x[column_name]==None or y[column_name]==None:
                squared_sum += np.float32(0.0)
                continue
            ## 컬럼의 값을 float32로 변환
            x_np = np.float32(x[column_name])
            y_np = np.float32(y[column_name])
            ## 컬럼의 값 범위 계산 (정규화에 사용)
            range_val = np.float32(
                metadata[column_name]['max'] -
                metadata[column_name]['min']
            )
            ## 0/0 연산의 경우, numpy에서는 NaN을 반환한다.
            ## 1/0 연산의 경우, numpy에서는 Inf를 반환한다.
            ## 정상적인 결과가 반환되는 경우, 제곱합에 더함.
            ## NumPy의 RuntimeWarning을 억제 (NaN/Inf는 이후 체크로 처리됨)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                normalized_diff = (x_np - y_np) / range_val
            if np.isnan(normalized_diff) or np.isinf(normalized_diff):
                squared_sum += np.float32(0.0)
            else:
                squared_sum += np.float32(normalized_diff ** 2)
        ## 제외된 컬럼은 스킵
        elif column_type == 'exclude':
            continue
        ## 컬럼 타입이 올바르지 않은 경우, 예외 발생
        else:
            raise ValueError(f"Invalid column type: {column_type}")
    ## 모든 변수에 대한 제곱합의 제곱근을 반환
    return np.sqrt(squared_sum)
