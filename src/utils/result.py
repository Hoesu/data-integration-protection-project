from datetime import datetime
from pathlib import Path

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
    with open(config_filename, "w") as f:
        yaml.dump(config, f)