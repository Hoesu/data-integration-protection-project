import logging
from pathlib import Path


def setup_logging(log_path: Path | None = None):
    """로깅 시스템을 설정하고 로거를 반환.

    파일 핸들러와 콘솔 핸들러를 모두 설정하여 로그를 파일과 콘솔에 동시에 출력합니다.
    기존 핸들러가 있으면 제거하고 새로운 핸들러를 추가합니다.

    Parameters
    ----------
    log_path : Path, optional
        로그 파일을 저장할 디렉토리 경로. None이면 기본값 "logs" 디렉토리를 사용합니다.
        기본값은 None입니다.

    Returns
    -------
    logging.Logger
        설정된 로거 객체. "project"라는 이름으로 등록됩니다.
    """
    ## 로그 디렉토리 설정: 인자로 전달받은 경로가 있으면 사용, 없으면 기본값 "logs" 사용
    log_dir = log_path if log_path else Path("logs")
    ## 로그 디렉토리 생성 (부모 디렉토리도 함께 생성, 이미 존재하면 무시)
    log_dir.mkdir(parents=True, exist_ok=True)
    ## 로그 파일 경로 생성
    log_filename = log_dir / "history.log"
    ## "project"라는 이름의 로거 가져오기
    logger = logging.getLogger("project")
    ## 로거 레벨을 DEBUG로 설정
    logger.setLevel(logging.DEBUG)
    ## 기존 핸들러 제거 (중복 핸들러 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    ## 로그 포맷터 설정: 시간, 로거 이름, 레벨, 메시지 포함
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    ## 파일 핸들러 생성 및 설정
    file_handler = logging.FileHandler(str(log_filename), encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    ## 콘솔 핸들러 생성 및 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    ## 로거에 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    ## 로그 파일 저장 위치를 로그로 기록
    logger.info(f"로그 파일 저장 위치: {log_filename}")
    return logger