import networkx as nx
import numpy as np


def calculate_identity_risk(
    graph: nx.Graph,
) -> np.ndarray:
    """
    각 노드에 대한 식별자 노출위험(IDR)을 계산합니다.

    식별자 노출위험은 특정 노드가 전체 네트워크 내에서 구조적 혹은 통계적으로
    고립된 형태를 보일 때 발생할 것으로 예상합니다.

    기본 아이디어는 한 노드가 주변과 얼마나 `약하게` 연결되어 있는지를 보는 것입니다.
    단순히 가중치 합 S_i만 보면,
    - 연결 2개(각 1.0)와 연결 20개(각 0.1)의 S_i가 같을 수 있어
      연결 수 차이가 반영되지 않는 문제가 있습니다.

    이를 보완하기 위해 **평균 가중치 기반** 정의를 사용합니다:

        IDR_i = 1 / (1 + S_i / k_i)

    여기서:
    - S_i : 노드 i에 연결된 모든 엣지의 가중치 합
    - k_i : 노드 i의 이웃 수

    해석:
    - 평균 가중치(S_i / k_i)가 클수록, 주변과 강하게 연결되어 있으므로 위험이 낮아짐
    - 평균 가중치가 작거나 연결이 없으면, 고립된 노드로 보고 위험이 높아짐
    
    Parameters
    ----------
    graph : nx.Graph
        가중치가 포함된 NetworkX 그래프 객체
        엣지의 가중치는 'weight' 속성에 저장되어 있어야 합니다.
        
    Returns
    -------
    np.ndarray
        각 노드에 대한 식별자 노출위험 값 (1차원 배열)
        배열의 인덱스는 노드 인덱스와 일치합니다.
        IDR_i 값이 높을수록 식별자 노출 위험이 높습니다.
        
    Examples
    --------
    >>> graph = nx.Graph()
    >>> graph.add_edge(0, 1, weight=0.8)
    >>> graph.add_edge(0, 2, weight=0.6)
    >>> idr = calculate_identity_risk(graph)
    >>> idr[0]  # 노드 0의 IDR
    0.4166...  # 1 / (1 + 1.4)
    """
    # 그래프의 노드 개수 확인
    num_nodes = graph.number_of_nodes()
    
    # 노드 개수만큼의 빈 배열 초기화 (IDR 값을 저장할 배열)
    idr_array = np.zeros(num_nodes, dtype=np.float64)
    
    # 노드 리스트를 정렬하여 일관된 인덱싱 보장
    node_list = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    
    # 각 노드 i에 대해:
    for node in node_list:
        idx = node_to_index[node]

        # graph[i] 또는 graph.neighbors(i)를 사용하여 이웃 노드들 확인
        neighbors = list(graph.neighbors(node))

        # 모든 이웃 노드의 가중치를 합산하여 S_i 계산
        S_i = 0.0
        for neighbor in neighbors:
            # 각 이웃 노드 j에 대해 엣지 (i, j)의 가중치 추출
            edge_weight = graph[node][neighbor].get('weight', 0.0)
            S_i += edge_weight

        # 연결 수 k_i 계산
        k_i = len(neighbors)

        # 평균 가중치 기반 IDR 계산
        # - 연결이 없으면(S_i=0, k_i=0) IDR = 1.0 (최대 식별 위험)
        # - 그 외에는 평균 가중치가 작을수록(연결은 많지만 약할수록) 위험이 높게 계산됨
        if k_i == 0:
            idr_array[idx] = 1.0
        else:
            avg_weight = S_i / k_i
            idr_array[idx] = 1.0 / (1.0 + avg_weight)
    
    # 계산된 IDR 배열 반환
    return idr_array

