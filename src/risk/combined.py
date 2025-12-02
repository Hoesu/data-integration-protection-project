import networkx as nx
import numpy as np
import pandas as pd

from .identity import calculate_identity_risk
from .attribute import calculate_attribute_risk


def _normalize_risks(idr: np.ndarray, adr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    IDR과 ADR을 0-1 범위로 정규화합니다.

    이론적 최소/최대값을 사용하여 두 위험도를 동일한 스케일로 변환합니다.
    - IDR: 이론적 범위 0 ~ 1.0 (이미 0~1 범위이므로 그대로 사용)
    - ADR: 이론적 범위 0.0 ~ 1.0 (이미 0~1 범위이므로 그대로 사용)

    이론적 범위를 사용하면 실제 데이터의 분포에 의존하지 않고
    일관된 정규화를 수행할 수 있습니다.

    Parameters
    ----------
    idr : np.ndarray
        식별자 노출위험 배열 (이론적 범위: 0 ~ 1.0)
    adr : np.ndarray
        속성 노출위험 배열 (이론적 범위: 0.0 ~ 1.0)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        정규화된 (IDR, ADR) 튜플 (둘 다 0~1 범위)
    """
    # IDR 정규화: 이론적 범위 0 ~ 1.0
    # 이미 0~1 범위이므로 그대로 사용 (클리핑만 수행)
    idr_normalized = np.clip(idr, 0.0, 1.0)

    # ADR 정규화: 이론적 범위 0.0 ~ 1.0
    # 이미 0~1 범위이므로 그대로 사용 (클리핑만 수행)
    adr_normalized = np.clip(adr, 0.0, 1.0)

    return idr_normalized, adr_normalized


def calculate_max_disclosure_risk(
    graph: nx.Graph,
    alpha: float = 1.0,
    beta: float = 1.0,
    method: str = 'max',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    각 노드에 대한 정규화된 IDR, ADR, 그리고 최종 노출위험(MDR)을 계산합니다.

    여러 방법을 통해 IDR과 ADR을 결합할 수 있습니다:

    1. 'max': 정규화 후 두 위험 중 더 높은 값 선택 (기본값)
       MDR_i = max(α * normalized(IDR_i), β * normalized(ADR_i))

    2. 'normalized_sum': 정규화 후 합산
       MDR_i = normalized(IDR_i) + normalized(ADR_i)

    3. 'normalized_weighted_avg': 정규화 후 가중 평균
       MDR_i = α * normalized(IDR_i) + β * normalized(ADR_i)

    4. 'geometric_mean': 기하 평균
       MDR_i = sqrt((α * IDR_i) * (β * ADR_i))

    Parameters
    ----------
    graph : nx.Graph
        가중치가 포함된 NetworkX 그래프 객체
    alpha : float, optional
        식별자 노출위험에 대한 가중치 (기본값: 1.0)
    beta : float, optional
        속성 노출위험에 대한 가중치 (기본값: 1.0)
    method : str, optional
        위험도 결합 방법: 'max', 'normalized_sum', 'normalized_weighted_avg', 'geometric_mean'
        (기본값: 'max')

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (정규화된 IDR, 정규화된 ADR, MDR) 튜플
        - 정규화된 IDR: 각 노드의 정규화된 식별자 노출위험 (0~1 범위)
        - 정규화된 ADR: 각 노드의 정규화된 속성 노출위험 (0~1 범위)
        - MDR: 각 노드의 최종 노출위험 값
        MDR_i 값이 높을수록 노출 위험이 높습니다.

    Examples
    --------
    >>> graph = nx.Graph()
    >>> graph.add_edge(0, 1, weight=0.8)
    >>> idr_norm, adr_norm, mdr = calculate_max_disclosure_risk(graph, method='normalized_sum')
    >>> mdr[0]
    0.xxx
    """
    # calculate_identity_risk(graph) 호출하여 IDR 배열 획득
    idr = calculate_identity_risk(graph)

    # calculate_attribute_risk(graph) 호출하여 ADR 배열 획득
    adr = calculate_attribute_risk(graph)

    # 정규화 수행 (이론적 범위 사용)
    idr_norm, adr_norm = _normalize_risks(idr, adr)

    # 방법에 따라 MDR 계산
    if method == 'max':
        # 정규화 후 두 위험 중 더 높은 값 선택
        weighted_idr_norm = alpha * idr_norm
        weighted_adr_norm = beta * adr_norm
        mdr = np.maximum(weighted_idr_norm, weighted_adr_norm)

    elif method == 'normalized_sum':
        # 정규화 후 합산: 두 위험을 동일한 스케일로 맞춘 후 합산
        mdr = idr_norm + adr_norm

    elif method == 'normalized_weighted_avg':
        # 정규화 후 가중 평균: 정규화 후 가중치 적용
        # 가중치 정규화 (합이 1이 되도록)
        total_weight = alpha + beta
        if total_weight > 0:
            alpha_norm = alpha / total_weight
            beta_norm = beta / total_weight
        else:
            alpha_norm = 0.5
            beta_norm = 0.5
        mdr = alpha_norm * idr_norm + beta_norm * adr_norm

    elif method == 'geometric_mean':
        # 기하 평균: 두 위험의 곱의 제곱근
        weighted_idr = alpha * idr
        weighted_adr = beta * adr
        # 음수 방지 및 0 처리
        weighted_idr = np.maximum(weighted_idr, 1e-10)
        weighted_adr = np.maximum(weighted_adr, 1e-10)
        mdr = np.sqrt(weighted_idr * weighted_adr)

    else:
        raise ValueError(
            f"지원하지 않는 방법: {method}. "
            "사용 가능한 방법: 'max', 'normalized_sum', 'normalized_weighted_avg', 'geometric_mean'"
        )

    # 정규화된 IDR, ADR, MDR 반환
    return idr_norm, adr_norm, mdr


def calculate_dataset_risk(
    mdr: np.ndarray,
    top_percent: float = 5.0,
) -> float:
    """
    데이터셋 단위 노출 위험을 계산합니다.

    데이터셋 단위 노출 위험은 개별 관측치 단위 노출 위험도의 상위 n%의
    평균값으로 측정합니다.

    Parameters
    ----------
    mdr : np.ndarray
        각 노드에 대한 최종 노출위험 값 배열
    top_percent : float, optional
        상위 몇 퍼센트를 사용할지 (기본값: 5.0, 즉 상위 5%)

    Returns
    -------
    float
        데이터셋 단위 노출 위험 값
        상위 n%의 MDR 값들의 평균

    Examples
    --------
    >>> mdr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    >>> dataset_risk = calculate_dataset_risk(mdr, top_percent=10.0)
    >>> dataset_risk
    0.95  # 상위 10% (1개)의 평균: 1.0
    """
    # mdr 배열이 비어있지 않은지 검증
    if len(mdr) == 0:
        raise ValueError("MDR 배열이 비어있습니다.")

    # top_percent가 0과 100 사이인지 검증
    if not (0.0 <= top_percent <= 100.0):
        raise ValueError(f"top_percent는 0과 100 사이의 값이어야 합니다. 현재 값: {top_percent}")

    # 상위 n%에 해당하는 개수 계산: n = len(mdr) * (top_percent / 100)
    n = int(len(mdr) * (top_percent / 100.0))

    # n이 0보다 작으면 1로 설정 (최소 1개)
    if n < 1:
        n = 1

    # mdr 배열을 내림차순으로 정렬
    sorted_mdr = np.sort(mdr)[::-1]

    # 상위 n개의 값 추출
    top_values = sorted_mdr[:n]

    # 추출된 값들의 평균 계산
    dataset_risk = float(np.mean(top_values))

    # 평균값 반환
    return dataset_risk


def calculate_risk(
    graph: nx.Graph,
    data: pd.DataFrame,
    config: dict,
) -> dict:
    """
    그래프 네트워크로부터 모든 노출 위험도를 계산합니다.

    식별자 노출위험, 속성 노출위험, 최종 노출위험, 그리고 데이터셋 단위 위험을
    모두 계산하여 반환합니다.

    Parameters
    ----------
    graph : nx.Graph
        구축된 그래프 네트워크
    data : pd.DataFrame
        원본 데이터프레임 (노드 인덱스와 행 인덱스가 일치해야 함)
    config : dict
        설정 딕셔너리
        - alpha: 식별자 노출위험 가중치 (기본값: 1.0)
        - beta: 속성 노출위험 가중치 (기본값: 1.0)
        - top_percent: 데이터셋 위험 계산 시 상위 퍼센트 (기본값: 5.0)
        - method: 위험도 결합 방법 (기본값: 'max')
            - 'max': 두 위험 중 더 높은 값 선택
            - 'normalized_sum': 정규화 후 합산
            - 'normalized_weighted_avg': 정규화 후 가중 평균
            - 'geometric_mean': 기하 평균

    Returns
    -------
    dict
        노출 위험도 정보를 담은 딕셔너리
        {
            'identity_risk_normalized': np.ndarray,  # 각 노드의 정규화된 IDR 값 (0~1)
            'attribute_risk_normalized': np.ndarray,  # 각 노드의 정규화된 ADR 값 (0~1)
            'max_disclosure_risk': np.ndarray, # 각 노드의 MDR 값
            'dataset_risk': float,            # 데이터셋 단위 위험
            'node_identifiers': np.ndarray,   # 각 노드의 식별자 (발급회원번호 등)
            'risk_by_row': pd.DataFrame,      # 행별 위험도 정보 (선택적, 정규화된 값만 포함)
        }

    Examples
    --------
    >>> risk_dict = calculate_risk(graph, data, config)
    >>> risk_dict['dataset_risk']
    0.85
    >>> risk_dict['max_disclosure_risk'][0]
    0.42
    """
    # config에서 파라미터 추출
    # alpha: 기본값 1.0
    alpha = config.get('risk', {}).get('alpha', 1.0)
    # beta: 기본값 1.0
    beta = config.get('risk', {}).get('beta', 1.0)
    # top_percent: 기본값 5.0
    top_percent = config.get('risk', {}).get('top_percent', 5.0)
    # method: 기본값 'max'
    method = config.get('risk', {}).get('method', 'max')

    # calculate_max_disclosure_risk 함수가 내부에서 IDR, ADR 계산 및 정규화를 수행
    # 정규화된 IDR, ADR, MDR을 한꺼번에 반환
    idr_normalized, adr_normalized, mdr = calculate_max_disclosure_risk(
        graph, alpha=alpha, beta=beta, method=method
    )

    # calculate_dataset_risk(mdr, top_percent) 호출하여 데이터셋 위험 계산
    dataset_risk = calculate_dataset_risk(mdr, top_percent=top_percent)

    # 그래프의 노드 리스트 가져오기 (정렬된 순서)
    node_list = sorted(graph.nodes())

    # 결과 딕셔너리 구성 (정규화된 값만 포함)
    result = {
        'identity_risk_normalized': idr_normalized,
        'attribute_risk_normalized': adr_normalized,
        'max_disclosure_risk': mdr,
        'dataset_risk': dataset_risk,
        'node_identifiers': np.array(node_list),  # 노드 식별자 추가
    }

    # risk_by_row: DataFrame (선택적, data와 위험도 정보를 결합)
    # 노드 순서와 데이터프레임 인덱스가 일치한다고 가정
    if len(data) == len(mdr):
        risk_by_row = data.copy()
        risk_by_row['identity_risk_normalized'] = idr_normalized
        risk_by_row['attribute_risk_normalized'] = adr_normalized
        risk_by_row['max_disclosure_risk'] = mdr
        result['risk_by_row'] = risk_by_row

    # 결과 딕셔너리 반환
    return result