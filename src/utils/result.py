from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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

def visualize_adjacency_matrix(
    adjacency_matrix: np.ndarray,
    result_dir: Path,
    config: dict
) -> None:
    """
    인접 행렬을 격자형 히트맵으로 시각화하고 저장.
    값이 클수록 진한 색, 작을수록 연한 색으로 표시.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        시각화할 인접 행렬.
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

    matrix_filename = result_dir / "adjacency_matrix.png"
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 값이 클수록 진한 색, 작을수록 연한 색을 위한 colormap 사용
    im = ax.imshow(
        adjacency_matrix,
        cmap='YlOrRd',  # Yellow-Orange-Red: 값이 클수록 진한 빨강
        aspect='auto',
        interpolation='nearest',
        origin='upper'
    )
    
    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Weight', rotation=270, labelpad=20)
    
    # 축 레이블 및 제목
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    ax.set_title('Adjacency Matrix Heatmap', fontsize=14, fontweight='bold')
    
    # 격자 표시 (선택적, 행렬이 크면 생략 가능)
    if adjacency_matrix.shape[0] <= 50:
        ax.set_xticks(np.arange(adjacency_matrix.shape[1]))
        ax.set_yticks(np.arange(adjacency_matrix.shape[0]))
        ax.set_xticklabels(np.arange(adjacency_matrix.shape[1]), fontsize=8)
        ax.set_yticklabels(np.arange(adjacency_matrix.shape[0]), fontsize=8)
    else:
        # 행렬이 크면 틱 간격을 넓게 설정
        tick_interval = max(1, adjacency_matrix.shape[0] // 20)
        ax.set_xticks(np.arange(0, adjacency_matrix.shape[1], tick_interval))
        ax.set_yticks(np.arange(0, adjacency_matrix.shape[0], tick_interval))
    
    plt.tight_layout()
    plt.savefig(matrix_filename, dpi=300, bbox_inches='tight')
    plt.close()

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