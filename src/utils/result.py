from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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
    config: dict,
    risk_results: dict = None
) -> None:
    """
    그래프를 시각화하고 저장.
    위험도 결과가 제공되면 세 가지 위험도 기준으로 각각 그래프를 생성합니다.

    Parameters
    ----------
    graph : nx.Graph
        시각화할 네트워크 그래프.
    result_dir : Path
        결과 저장 디렉토리 경로.
    config : dict
        설정 딕셔너리 (현재 미사용, 향후 확장 가능).
    risk_results : dict, optional
        위험도 계산 결과 딕셔너리. 제공되면 위험도에 따라 노드 색상을 매핑합니다.
        {
            'identity_risk_normalized': np.ndarray,
            'attribute_risk_normalized': np.ndarray,
            'max_disclosure_risk': np.ndarray,
        }
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

    # 레이아웃 계산 (엣지 가중치 고려) - 한 번만 계산하여 재사용
    try:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    except Exception:
        pos = nx.spring_layout(graph)

    # 엣지 가중치 안전하게 추출
    edge_widths = []
    for u, v in graph.edges():
        weight = graph[u][v].get('weight', 1.0)
        edge_widths.append(max(0.5, weight * 2))

    # 위험도 결과가 있으면 세 가지 그래프 생성
    if risk_results is not None:
        risk_types = [
            ('identity_risk_normalized', '식별자 노출위험 (IDR)', 'graph_idr.png'),
            ('attribute_risk_normalized', '속성 노출위험 (ADR)', 'graph_adr.png'),
            ('max_disclosure_risk', '최종 노출위험 (MDR)', 'graph_mdr.png')
        ]

        for risk_key, risk_title, filename in risk_types:
            if risk_key not in risk_results:
                continue

            risk_values = risk_results[risk_key]
            
            # 위험도 계산 시 정렬된 노드 리스트를 사용하므로, 여기서도 정렬된 순서 사용
            # node_identifiers가 있으면 사용, 없으면 sorted(graph.nodes()) 사용
            if 'node_identifiers' in risk_results:
                sorted_node_list = list(risk_results['node_identifiers'])
            else:
                sorted_node_list = sorted(graph.nodes())
            
            # 정렬된 노드 리스트와 위험도 값 매핑 (인덱스 기반)
            node_risk_map = {
                node: risk_values[i] 
                for i, node in enumerate(sorted_node_list) 
                if i < len(risk_values)
            }
            
            # 그래프의 실제 노드 순서에 맞춰 위험도 값 배열 생성
            graph_node_list = list(graph.nodes())
            node_colors = [node_risk_map.get(node, 0.0) for node in graph_node_list]

            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.axis('off')

            # 위험도 값에 따라 색상 매핑 (초록색(낮음) -> 빨간색(높음))
            # RdYlGn_r: Red-Yellow-Green reversed (빨강=높음, 초록=낮음)
            cmap = plt.cm.RdYlGn_r
            vmin = 0.0
            vmax = 1.0

            # 노드 그리기
            nodes = nx.draw_networkx_nodes(
                graph, pos,
                node_size=300,
                node_color=node_colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                linewidths=1.5,
                edgecolors='darkblue',
                ax=ax
            )

            # 엣지 그리기
            nx.draw_networkx_edges(
                graph, pos,
                width=edge_widths,
                edge_color='gray',
                alpha=0.6,
                style='solid',
                ax=ax
            )

            # 레이블 그리기
            nx.draw_networkx_labels(
                graph, pos,
                font_size=6,
                font_weight='bold',
                font_family=plt.rcParams['font.family'],
                ax=ax
            )

            # 컬러바 추가
            if nodes is not None:
                cbar = plt.colorbar(nodes, ax=ax)
                cbar.set_label(risk_title, rotation=270, labelpad=20)

            ax.set_title(f'Graph Visualization - {risk_title}', fontsize=14, fontweight='bold')

            plt.tight_layout()
            graph_filename = result_dir / filename
            plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
            plt.close()

    # 기본 그래프도 생성 (위험도 정보 없이)
    graph_filename = result_dir / "graph.png"
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    nx.draw_networkx_nodes(
        graph, pos,
        node_size=300,
        node_color='lightblue',
        alpha=0.9,
        linewidths=1.5,
        edgecolors='darkblue',
        ax=ax
    )

    nx.draw_networkx_edges(
        graph, pos,
        width=edge_widths,
        edge_color='gray',
        alpha=0.6,
        style='solid',
        ax=ax
    )

    nx.draw_networkx_labels(
        graph, pos,
        font_size=6,
        font_weight='bold',
        font_family=plt.rcParams['font.family'],
        ax=ax
    )

    ax.set_title('Graph Visualization (Default)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_risk_results(
    risk_results: dict,
    result_dir: Path,
) -> None:
    """
    위험도 계산 결과를 파일로 저장합니다.

    Parameters
    ----------
    risk_results : dict
        위험도 계산 결과 딕셔너리
        {
            'identity_risk_normalized': np.ndarray,  # 정규화된 IDR (0~1)
            'attribute_risk_normalized': np.ndarray,  # 정규화된 ADR (0~1)
            'max_disclosure_risk': np.ndarray,  # MDR
            'dataset_risk': float,
            'node_identifiers': np.ndarray (선택적),
            'risk_by_row': pd.DataFrame (선택적),
        }
    result_dir : Path
        결과 저장 디렉토리 경로.
    """
    # 위험도 요약 정보를 텍스트 파일로 저장
    summary_filename = result_dir / "risk_summary.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("위험도 계산 결과 요약\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"데이터셋 전체 위험도: {risk_results['dataset_risk']:.6f}\n\n")

        # 정규화된 값만 사용 (원본 값은 더 이상 저장하지 않음)
        idr_norm = risk_results['identity_risk_normalized']
        adr_norm = risk_results['attribute_risk_normalized']
        mdr = risk_results['max_disclosure_risk']

        f.write("식별자 노출위험 (IDR, 정규화됨) 통계:\n")
        f.write(f"  평균: {np.mean(idr_norm):.6f}\n")
        f.write(f"  표준편차: {np.std(idr_norm):.6f}\n")
        f.write(f"  최소값: {np.min(idr_norm):.6f}\n")
        f.write(f"  최대값: {np.max(idr_norm):.6f}\n")
        f.write(f"  중앙값: {np.median(idr_norm):.6f}\n\n")

        f.write("속성 노출위험 (ADR, 정규화됨) 통계:\n")
        f.write(f"  평균: {np.mean(adr_norm):.6f}\n")
        f.write(f"  표준편차: {np.std(adr_norm):.6f}\n")
        f.write(f"  최소값: {np.min(adr_norm):.6f}\n")
        f.write(f"  최대값: {np.max(adr_norm):.6f}\n")
        f.write(f"  중앙값: {np.median(adr_norm):.6f}\n\n")

        f.write("최종 노출위험 (MDR) 통계:\n")
        f.write(f"  평균: {np.mean(mdr):.6f}\n")
        f.write(f"  표준편차: {np.std(mdr):.6f}\n")
        f.write(f"  최소값: {np.min(mdr):.6f}\n")
        f.write(f"  최대값: {np.max(mdr):.6f}\n")
        f.write(f"  중앙값: {np.median(mdr):.6f}\n\n")

        # 상위 위험 노드 정보
        top_n = min(10, len(mdr))
        top_indices = np.argsort(mdr)[::-1][:top_n]
        f.write(f"상위 {top_n}개 위험 노드:\n")
        f.write("-" * 60 + "\n")
        # 노드 식별자가 있으면 포함
        if 'node_identifiers' in risk_results:
            node_ids = risk_results['node_identifiers']
            f.write(f"{'순위':<6} {'노드ID':<20} {'IDR(norm)':<12} {'ADR(norm)':<12} {'MDR':<12}\n")
            f.write("-" * 70 + "\n")
            for rank, idx in enumerate(top_indices, 1):
                f.write(f"{rank:<6} {str(node_ids[idx]):<20} {idr_norm[idx]:<12.6f} {adr_norm[idx]:<12.6f} {mdr[idx]:<12.6f}\n")
        else:
            f.write(f"{'순위':<6} {'노드인덱스':<12} {'IDR(norm)':<12} {'ADR(norm)':<12} {'MDR':<12}\n")
            f.write("-" * 70 + "\n")
            for rank, idx in enumerate(top_indices, 1):
                f.write(f"{rank:<6} {idx:<12} {idr_norm[idx]:<12.6f} {adr_norm[idx]:<12.6f} {mdr[idx]:<12.6f}\n")

    # 위험도 배열을 CSV로 저장 (정규화된 값만 포함)
    risk_array_filename = result_dir / "risk_values.csv"
    # 노드 식별자가 있으면 포함
    if 'node_identifiers' in risk_results:
        risk_df = pd.DataFrame({
            'node_identifier': risk_results['node_identifiers'],
            'identity_risk_normalized': risk_results['identity_risk_normalized'],
            'attribute_risk_normalized': risk_results['attribute_risk_normalized'],
            'max_disclosure_risk': risk_results['max_disclosure_risk'],
        })
    else:
        # 노드 식별자가 없으면 인덱스 사용
        risk_df = pd.DataFrame({
            'node_index': range(len(risk_results['identity_risk_normalized'])),
            'identity_risk_normalized': risk_results['identity_risk_normalized'],
            'attribute_risk_normalized': risk_results['attribute_risk_normalized'],
            'max_disclosure_risk': risk_results['max_disclosure_risk'],
        })

    # 수치형 컬럼은 소수 넷째 자리에서 반올림
    numeric_cols = [col for col in risk_df.columns if col != 'node_identifier' and col != 'node_index']
    risk_df[numeric_cols] = risk_df[numeric_cols].round(4)

    risk_df.to_csv(risk_array_filename, index=False, encoding='utf-8')

    # risk_by_row가 있으면 저장
    if 'risk_by_row' in risk_results and risk_results['risk_by_row'] is not None:
        risk_by_row_filename = result_dir / "risk_by_row.csv"
        risk_results['risk_by_row'].to_csv(risk_by_row_filename, index=False, encoding='utf-8')