import networkx as nx
import numpy as np


def calculate_identity_risk(
    graph: nx.Graph,
) -> np.ndarray:
    """
    각 노드에 대한 식별자 노출위험(IDR)을 계산합니다.

    식별자 노출위험은 특정 노드가 전체 네트워크 내에서 구조적으로
    약하게 연결되어 있을 때 높아집니다. 연결이 약하거나 고립된 노드일수록
    식별자가 노출될 위험이 크다고 판단합니다.

    알고리즘:
    ---------
    1. 각 노드의 평균 엣지 가중치를 계산: avg_weight_i = S_i / k_i
       - S_i: 노드 i에 연결된 모든 엣지의 가중치 합
       - k_i: 노드 i의 이웃 수 (차수)
    
    2. 평균 가중치를 기준으로 모든 노드의 순위를 매김 (오름차순)
       - 낮은 평균 가중치 → 낮은 순위 (rank ≈ 0)
       - 높은 평균 가중치 → 높은 순위 (rank ≈ N-1)
    
    3. 백분위수 기반 IDR로 변환:
       
       IDR_i = 1 - (rank_i / (N - 1))
       
       - 약한 연결 (낮은 순위) → 높은 IDR (위험 높음)
       - 강한 연결 (높은 순위) → 낮은 IDR (위험 낮음)
       - 고립 노드 (k_i = 0) → IDR = 1.0 (최대 위험)

    특징:
    -----
    - 0~1 사이 균등 분포: 백분위수 변환으로 값이 고르게 분포
    - 상대적 평가: 현재 그래프 내에서 상대적인 연결 강도를 반영
    - 연결 수와 가중치 모두 고려: 평균 가중치로 두 요소를 통합

    값의 해석:
    ---------
    - IDR ≈ 1.0: 고립 노드 또는 가장 약하게 연결된 노드 (상위 10%)
    - IDR ≈ 0.5: 중간 수준의 연결 강도를 가진 노드
    - IDR ≈ 0.0: 가장 강하게 연결된 노드 (하위 10%)

    Parameters
    ----------
    graph : nx.Graph
        가중치가 포함된 NetworkX 그래프 객체.
        엣지의 가중치는 'weight' 속성에 저장되어 있어야 합니다.

    Returns
    -------
    np.ndarray
        각 노드에 대한 식별자 노출위험 값 (1차원 배열).
        배열의 인덱스는 정렬된 노드 인덱스와 일치합니다.
        값의 범위는 [0.0, 1.0]이며, 높을수록 위험이 큽니다.

    Examples
    --------
    >>> import networkx as nx
    >>> graph = nx.Graph()
    >>> graph.add_edge(0, 1, weight=0.8)
    >>> graph.add_edge(1, 2, weight=0.6)
    >>> graph.add_edge(2, 3, weight=0.4)
    >>> graph.add_node(4)  # 고립 노드
    >>> idr = calculate_identity_risk(graph)
    >>> idr[4]  # 고립 노드는 항상 1.0
    1.0
    >>> idr[0] < idr[3]  # 강한 연결(0) < 약한 연결(3)
    True
    """
    num_nodes = graph.number_of_nodes()
    idr_array = np.zeros(num_nodes, dtype=np.float64)
    
    # 노드를 정렬하여 일관된 인덱싱 보장
    node_list = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # 1단계: 각 노드의 평균 엣지 가중치 계산
    avg_weights = np.full(num_nodes, -np.inf)
    
    for node in node_list:
        idx = node_to_index[node]
        neighbors = list(graph.neighbors(node))
        
        if len(neighbors) == 0:
            # 고립 노드는 -inf로 표시 (최하위 순위)
            continue
        
        # 평균 가중치 = 가중치 합 / 이웃 수
        weight_sum = sum(graph[node][neighbor].get('weight', 0.0) 
                        for neighbor in neighbors)
        avg_weights[idx] = weight_sum / len(neighbors)
    
    # 2단계: 평균 가중치 기준 오름차순 순위 계산
    # argsort를 2번 적용하여 각 원소의 순위를 계산
    ranks = np.argsort(np.argsort(avg_weights))
    
    # 3단계: 순위를 백분위수 기반 IDR로 변환
    for idx in range(num_nodes):
        if avg_weights[idx] == -np.inf:
            # 고립 노드는 최대 위험
            idr_array[idx] = 1.0
        else:
            # IDR = 1 - (순위 백분위수)
            if num_nodes > 1:
                idr_array[idx] = 1.0 - (ranks[idx] / (num_nodes - 1))
            else:
                idr_array[idx] = 0.5

    return idr_array

