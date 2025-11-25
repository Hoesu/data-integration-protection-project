from src.database import *
from src.utils import *
import argparse

if __name__ == "__main__":

    # 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    # 로깅 설정
    logger = setup_logging()

    # 설정 파일 로드
    config = load_config(args.config)
    logger.info(f"Config loaded: {config}")

    # 엔진 생성 및 테이블 생성
    engine = Base.get_engine()
    Base.metadata.create_all(engine)

    # 쿼리 실행
    result = execute_query(engine, config)