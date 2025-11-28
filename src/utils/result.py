from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import yaml


def setup_result_directory(config: dict) -> Path:
    """
    결과 디렉토리를 생성하고 반환.

    config의 action에 따라 data 또는 experiment 모드로 디렉토리를 생성.

    Parameters
    ----------
    config : dict
        설정 딕셔너리. 'data' 키에 'action', 'target_table', 'source_table' 포함.

    Returns
    -------
    Path
        생성된 결과 디렉토리 경로.
    """
    results_dir = Path.cwd() / "results"
    action_taken = config['data']['action']
    mode = 'data' if action_taken == 'insert' else 'experiment'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == 'data':
        table_name = config['data']['target_table']
        final_dir = results_dir / f"data_{table_name}_{timestamp}"
    else:
        table_name = config['data']['source_table']
        final_dir = results_dir / f"experiment_{table_name}_{timestamp}"
    final_dir.mkdir(parents=True, exist_ok=True)
    return final_dir

def save_config(config: dict, result_dir: Path) -> None:
    """
    설정을 YAML 파일로 저장.

    Parameters
    ----------
    config : dict
        저장할 설정 딕셔너리.
    result_dir : Path
        결과 디렉토리 경로.
    """
    config_filename = result_dir / "config.yaml"
    with open(config_filename, encoding='utf-8', mode="w") as f:
        yaml.dump(config, stream=f, allow_unicode=True)

def visualize_graph(
    graph: nx.Graph,
    result_dir: Path,
    config: dict
) -> None:
    """
    그래프를 시각화하고 저장.

    Parameters
    ----------
    graph : nx.Graph
        시각화할 네트워크 그래프.
    result_dir : Path
        결과 저장 디렉토리 경로.
    config : dict
        설정 딕셔너리 (현재 미사용, 향후 확장 가능).
    """
    # 한글 폰트 설정
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        # Linux에서 한글 폰트 시도
        import matplotlib.font_manager as fm
        font_list = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR', 'Malgun Gothic']
        for font in korean_fonts:
            if font in font_list:
                plt.rcParams['font.family'] = font
                break
    except Exception:
        pass  # 폰트 설정 실패 시 기본 폰트 사용

    graph_filename = result_dir / "graph.png"

    # 레이아웃 계산 (엣지 가중치 고려)
    try:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    except Exception:
        pos = nx.spring_layout(graph)

    plt.figure(figsize=(12, 10))
    plt.axis('off')

    # 엣지 가중치 안전하게 추출
    edge_widths = []
    for u, v in graph.edges():
        weight = graph[u][v].get('weight', 1.0)
        edge_widths.append(max(0.5, weight * 2))

    # 그래프 그리기
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=800,
        node_color='lightblue',
        alpha=0.9,
        linewidths=1.5,
        edgecolors='darkblue'
    )

    nx.draw_networkx_edges(
        graph, pos,
        width=edge_widths,
        edge_color='gray',
        alpha=0.6,
        style='solid'
    )

    nx.draw_networkx_labels(
        graph, pos,
        font_size=10,
        font_weight='bold',
        font_family=plt.rcParams['font.family']
    )

    plt.tight_layout()
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
    plt.close()