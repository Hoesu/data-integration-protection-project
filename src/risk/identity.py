import networkx as nx
import numpy as np


def calculate_identity_risk(
    graph: nx.Graph,
) -> np.ndarray:
    """
    각 노드에 대한 식별자 노출위험(IDR)을 계산합니다.
    
    식별자 노출위험은 특정 노드가 전체 네트워크 내에서 구조적 혹은 통계적으로
    고립된 형태를 보일 때 발생할 것으로 예상합니다.
    
    이는 weighted degree centrality의 역수로 표현됩니다:
    IDR_i = 1 / (1 + S_i)
    
    여기서 S_i는 노드 i에 연결된 모든 엣지의 가중치 합입니다.
    S_i가 낮을수록 고립되었음을 의미하므로 식별자 노출 위험이 높아집니다.
    
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
    # TODO: 그래프의 노드 개수 확인
    # TODO: 노드 개수만큼의 빈 배열 초기화 (IDR 값을 저장할 배열)
    # TODO: 각 노드 i에 대해:
    #   - graph[i] 또는 graph.neighbors(i)를 사용하여 이웃 노드들 확인
    #   - 각 이웃 노드 j에 대해 엣지 (i, j)의 가중치 추출 (graph[i][j]['weight'])
    #   - 모든 이웃 노드의 가중치를 합산하여 S_i 계산
    #   - IDR_i = 1 / (1 + S_i) 공식 적용
    #   - 배열의 i번째 위치에 IDR_i 값 저장
    # TODO: 계산된 IDR 배열 반환
    pass

