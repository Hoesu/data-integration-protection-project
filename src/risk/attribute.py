import networkx as nx
import numpy as np


def calculate_attribute_risk(
    graph: nx.Graph,
) -> np.ndarray:
    """
    각 노드에 대한 속성 노출위험(ADR)을 계산합니다.
    
    속성 노출위험은 개별 식별자는 다르더라도, 다수의 노드가 매우 높은 속성
    유사성을 공유하여 특정 속성 조합이 노출될 경우 나머지 속성까지 추론
    가능성이 높아질 때 증가할 것으로 예상합니다.
    
    이는 weighted clustering coefficient를 활용하여 계산됩니다.

    구현에서는 표준 weighted clustering coefficient 정의와 유사하게,
    다음과 같이 **w_jk 항을 제거한** 형태를 사용합니다:

        ADR_i = 1 / (S_i(k_i-1)) * sum_{j≠i} sum_{k≠i,j} ((w_ij + w_ik)/2 * a_jk)

    이렇게 하면:
    - i와 이웃들(j, k) 간의 연결 강도 (w_ij, w_ik)를 중심으로
    - j-k 간에 엣지가 존재하는지 여부(a_jk)만 반영하여
    - i 주변의 삼각관계 밀집도를 안정적으로 측정할 수 있습니다.
    
    여기서:
    - k_i: 노드 i의 이웃 수
    - S_i: 노드 i에 연결된 모든 엣지 강도의 합
    - w_ij: 노드 i와 j 간의 엣지 가중치
    - a_jk: 노드 j와 k가 연결되어 있으면 1, 아니면 0 (인접 행렬 원소)
    - w_jk: 노드 j와 k 간의 엣지 가중치 (연결되어 있는 경우)
    
    즉, 분석의 대상이 되는 노드와 삼각관계를 이루고 있는 노드들 간의
    연결된 정도를 모두 합산하고 정규화함으로써 해당 노드 주변이 얼마나
    밀집되어 있는가를 평가할 수 있습니다.
    
    Parameters
    ----------
    graph : nx.Graph
        가중치가 포함된 NetworkX 그래프 객체
        엣지의 가중치는 'weight' 속성에 저장되어 있어야 합니다.
        
    Returns
    -------
    np.ndarray
        각 노드에 대한 속성 노출위험 값 (1차원 배열)
        배열의 인덱스는 노드 인덱스와 일치합니다.
        ADR_i 값이 높을수록 속성 노출 위험이 높습니다.
        
    Examples
    --------
    >>> graph = nx.Graph()
    >>> graph.add_edge(0, 1, weight=0.8)
    >>> graph.add_edge(0, 2, weight=0.6)
    >>> graph.add_edge(1, 2, weight=0.7)  # 삼각관계 형성
    >>> adr = calculate_attribute_risk(graph)
    >>> adr[0]  # 노드 0의 ADR
    0.xxx  # 계산된 값
    """
    # 그래프의 노드 개수 확인
    num_nodes = graph.number_of_nodes()
    
    # 노드 개수만큼의 빈 배열 초기화 (ADR 값을 저장할 배열)
    adr_array = np.zeros(num_nodes, dtype=np.float64)
    
    # 노드 리스트를 정렬하여 일관된 인덱싱 보장
    node_list = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    
    # 각 노드 i에 대해:
    for node in node_list:
        idx = node_to_index[node]
        
        # 노드 i의 이웃 노드 리스트 추출
        neighbors = list(graph.neighbors(node))
        
        # k_i = 이웃 노드의 개수 계산
        k_i = len(neighbors)
        
        # S_i 계산: 노드 i에 연결된 모든 엣지의 가중치 합
        S_i = 0.0
        for neighbor in neighbors:
            edge_weight = graph[node][neighbor].get('weight', 0.0)
            S_i += edge_weight
        
        # k_i가 1 이하인 경우 ADR_i = 0으로 설정하고 다음 노드로 (삼각관계 불가능)
        if k_i <= 1:
            adr_array[idx] = 0.0
            continue
        
        # 이웃 노드 쌍 (j, k)에 대해 (j ≠ i, k ≠ i, j ≠ k):
        # 개선된 수식: 표준 weighted clustering coefficient에 맞게 w_jk를 제거
        # ADR_i ∝ sum_{j≠i} sum_{k≠i,j} ((w_ij + w_ik)/2 * a_jk)
        # (w_ij + w_ik)/2는 대칭이므로 j < k로 제한해도 결과는 동일 (효율성 향상)
        cumulative_sum = 0.0
        for idx_j, j in enumerate(neighbors):
            for k in neighbors[idx_j + 1:]:  # j < k를 보장하여 중복 계산 방지
                
                # 노드 j와 k가 연결되어 있는지 확인 (a_jk = 1 if connected, 0 otherwise)
                if graph.has_edge(j, k):
                    # w_ij: 노드 i와 j 간의 엣지 가중치
                    w_ij = graph[node][j].get('weight', 0.0)
                    # w_ik: 노드 i와 k 간의 엣지 가중치
                    w_ik = graph[node][k].get('weight', 0.0)
                    # 수식: (w_ij + w_ik) / 2 * a_jk, 여기서 a_jk = 1
                    cumulative_sum += (w_ij + w_ik) / 2.0
                # 연결되어 있지 않다면 a_jk = 0이므로 스킵
        
        # ADR_i = (누적된 합) / (S_i * (k_i - 1)) 계산
        # 분모가 0인 경우 처리 필요
        denominator = S_i * (k_i - 1)
        if denominator > 0:
            adr_array[idx] = cumulative_sum / denominator
        else:
            adr_array[idx] = 0.0
    
    # 계산된 ADR 배열 반환
    return adr_array

