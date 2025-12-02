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

    구현에서는 밀집도(density)와 규모(이웃 수)를 모두 고려합니다:

        1. 밀집도 계산: density_i = sum_{j<k} ((w_ij + w_ik)/2 * a_jk) / sum_{j<k} ((w_ij + w_ik)/2)
        2. 규모 스케일링: scale_i = sqrt(k_i / max_k)  (로그 스케일링 사용)
        3. 최종 ADR: ADR_i = density_i * scale_i

    이렇게 하면:
    - 밀집도가 높을수록 위험 증가 (삼각관계가 많을수록)
    - 이웃 수가 많을수록 위험 증가 (더 많은 노드와 속성 유사성 공유)
    - 두 요소를 곱하여 종합적인 위험도 측정

    여기서:
    - k_i: 노드 i의 이웃 수
    - max_k: 그래프 내 모든 노드의 최대 이웃 수
    - w_ij: 노드 i와 j 간의 엣지 가중치
    - a_jk: 노드 j와 k가 연결되어 있으면 1, 아니면 0

    밀집도 계산:
    - 분자: 실제 삼각관계를 형성하는 이웃 쌍들의 가중치 합
    - 분모: 가능한 모든 이웃 쌍들의 가중치 합
    - 결과: 0~1 범위의 밀집도 비율

    규모 스케일링:
    - 이웃 수가 많을수록 위험이 증가하지만, 로그 스케일링으로 과도한 증가 방지
    - sqrt(k_i / max_k)를 사용하여 0~1 범위 유지

    이론적 범위:
    - 최소값: 0.0 (k_i <= 1이거나 모든 이웃 쌍이 연결되지 않은 경우)
    - 최대값: 1.0 (완전 그래프이고 최대 이웃 수를 가진 경우)
    - 일반 범위: 0.0 <= ADR_i <= 1.0
    - 참고: 정규화는 calculate_max_disclosure_risk 함수에서 수행됩니다.

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

    # 노드 리스트를 정렬하여 일관된 인덱싱 보장
    node_list = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # 첫 번째 순회: 모든 노드의 이웃 수를 수집하여 최대값 찾기
    max_degree = 0
    for node in node_list:
        degree = graph.degree(node)
        if degree > max_degree:
            max_degree = degree

    # 최대 이웃 수가 0이거나 1 이하면 모든 노드의 ADR을 0으로 설정
    if max_degree <= 1:
        return np.zeros(num_nodes, dtype=np.float64)

    # 노드 개수만큼의 빈 배열 초기화 (ADR 값을 저장할 배열)
    adr_array = np.zeros(num_nodes, dtype=np.float64)

    # 두 번째 순회: 각 노드에 대해 ADR 계산
    for node in node_list:
        idx = node_to_index[node]

        # 노드 i의 이웃 노드 리스트 추출
        neighbors = list(graph.neighbors(node))

        # k_i = 이웃 노드의 개수 계산
        k_i = len(neighbors)

        # k_i가 1 이하인 경우 ADR_i = 0으로 설정하고 다음 노드로 (삼각관계 불가능)
        if k_i <= 1:
            adr_array[idx] = 0.0
            continue

        # 1단계: 밀집도 계산 (실제 삼각관계 / 가능한 삼각관계)
        # density_i = sum_{j<k} ((w_ij + w_ik)/2 * a_jk) / sum_{j<k} ((w_ij + w_ik)/2)
        numerator = 0.0  # 실제 삼각관계의 가중치 합
        denominator = 0.0  # 가능한 모든 삼각관계의 가중치 합

        for idx_j, j in enumerate(neighbors):
            w_ij = graph[node][j].get('weight', 0.0)
            for k in neighbors[idx_j + 1:]:  # j < k를 보장하여 중복 계산 방지
                w_ik = graph[node][k].get('weight', 0.0)

                # 가능한 모든 삼각관계의 가중치 합 (분모)
                weight_sum = (w_ij + w_ik) / 2.0
                denominator += weight_sum

                # 실제 삼각관계의 가중치 합 (분자)
                if graph.has_edge(j, k):
                    numerator += weight_sum

        # 밀집도 계산
        if denominator > 0:
            density = numerator / denominator
        else:
            density = 0.0

        # 2단계: 규모 스케일링 (이웃 수 고려)
        # sqrt(k_i / max_k)를 사용하여 0~1 범위 유지
        scale = np.sqrt(k_i / max_degree) if max_degree > 0 else 0.0

        # 3단계: 최종 ADR = 밀집도 × 규모
        adr_array[idx] = density * scale

    # 계산된 ADR 배열 반환
    return adr_array

