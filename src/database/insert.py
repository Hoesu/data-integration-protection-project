"""CSV 파일을 읽어서 CardUserInfo 테이블에 대용량 데이터를 안전하게 삽입하는 스크립트"""

import logging
from pathlib import Path
from typing import Union

import pandas as pd
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from .tables.base import Base, get_engine
from .tables.card_user_info import CardUserInfo

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_csv_to_dataframe(csv_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    CSV 파일을 읽어서 DataFrame으로 반환
    
    Args:
        csv_path: CSV 파일 경로
        **kwargs: pandas.read_csv에 전달할 추가 인자
    
    Returns:
        DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    
    logger.info(f"CSV 파일 읽기 시작: {csv_path}")
    df = pd.read_csv(csv_path, **kwargs)
    logger.info(f"CSV 파일 읽기 완료: {len(df):,}행")
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame을 데이터베이스 삽입에 적합한 형태로 전처리
    
    Args:
        df: 원본 DataFrame
    
    Returns:
        전처리된 DataFrame
    """
    df = df.copy()
    
    # CardUserInfo 모델에서 정수 컬럼 목록 가져오기
    from .tables.card_user_info import CardUserInfo
    from sqlalchemy import Integer
    
    # 정수 컬럼 목록 추출
    integer_columns = []
    for column_name, column in CardUserInfo.__table__.columns.items():
        if isinstance(column.type, Integer):
            integer_columns.append(column_name)
    
    # 각 컬럼 처리
    for col in df.columns:
        if col in integer_columns:
            # 정수 컬럼: NaN을 None으로 변환하고, float 타입이지만 정수 값인 경우 정수로 변환
            df[col] = df[col].apply(
                lambda x: None if pd.isna(x) else (int(x) if isinstance(x, float) else x)
            )
        elif df[col].dtype == 'object':
            # 문자열 컬럼에서 "_"를 None으로 변환
            df[col] = df[col].replace("_", None)
            # NaN도 None으로
            df[col] = df[col].where(pd.notna(df[col]), None)
        elif pd.api.types.is_float_dtype(df[col]):
            # float 컬럼: NaN을 None으로
            df[col] = df[col].where(pd.notna(df[col]), None)
    
    return df


def dataframe_to_dict_list(df: pd.DataFrame) -> list[dict]:
    """
    DataFrame을 딕셔너리 리스트로 변환
    
    Args:
        df: DataFrame
    
    Returns:
        딕셔너리 리스트
    """
    # CardUserInfo 모델에서 정수 컬럼 목록 가져오기
    from .tables.card_user_info import CardUserInfo
    from sqlalchemy import Integer
    
    # 정수 컬럼 목록 추출
    integer_columns = []
    for column_name, column in CardUserInfo.__table__.columns.items():
        if isinstance(column.type, Integer):
            integer_columns.append(column_name)
    
    # 딕셔너리로 변환
    records = df.to_dict('records')
    
    # 각 레코드에서 모든 NaN을 None으로 변환하고 정수 컬럼 처리
    for record in records:
        for col, value in record.items():
            # 모든 NaN을 None으로 변환
            if pd.isna(value):
                record[col] = None
            # 정수 컬럼이고 float 타입인 경우 정수로 변환
            elif col in integer_columns and isinstance(value, float):
                record[col] = int(value)
    
    return records


def bulk_insert_card_user_info(
    csv_path: Union[str, Path],
    batch_size: int = 1000,
    skip_duplicates: bool = True,
    create_table: bool = False,
    **csv_kwargs
) -> int:
    """
    CSV 파일을 읽어서 CardUserInfo 테이블에 대용량 데이터를 배치 단위로 삽입
    
    Args:
        csv_path: CSV 파일 경로
        batch_size: 배치 크기 (한 번에 삽입할 레코드 수)
        skip_duplicates: 중복 키가 있을 경우 건너뛰기 (True: ON CONFLICT DO NOTHING, False: 에러 발생)
        create_table: 테이블이 없을 경우 생성할지 여부
        **csv_kwargs: pandas.read_csv에 전달할 추가 인자
    
    Returns:
        삽입된 레코드 수
    """
    # CSV 파일 읽기
    df = read_csv_to_dataframe(csv_path, **csv_kwargs)
    
    # 전처리
    df = preprocess_dataframe(df)
    
    # 딕셔너리 리스트로 변환
    records = dataframe_to_dict_list(df)
    total_records = len(records)
    
    logger.info(f"총 {total_records:,}개의 레코드를 삽입합니다.")
    
    # 데이터베이스 연결
    engine = get_engine()
    
    # 테이블 생성 (필요한 경우)
    if create_table:
        logger.info("테이블 생성 중...")
        Base.metadata.create_all(engine)
        logger.info("테이블 생성 완료")
    
    # 세션 생성
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    inserted_count = 0
    error_count = 0
    
    try:
        # 배치 단위로 삽입
        with tqdm(total=total_records, desc="데이터 삽입 중", unit="행") as pbar:
            for i in range(0, total_records, batch_size):
                batch = records[i:i + batch_size]
                
                try:
                    # bulk_insert_mappings 사용 (ORM 객체 생성 없이 직접 삽입)
                    session.bulk_insert_mappings(
                        CardUserInfo,
                        batch,
                        render_nulls=True  # None 값을 NULL로 명시적으로 렌더링
                    )
                    session.commit()
                    inserted_count += len(batch)
                    pbar.update(len(batch))
                    
                except Exception as e:
                    session.rollback()
                    error_count += len(batch)
                    logger.error(f"배치 삽입 실패 (행 {i+1}~{i+len(batch)}): {e}")
                    
                    # 중복 키 에러인 경우 skip_duplicates 옵션에 따라 처리
                    if skip_duplicates and "duplicate key" in str(e).lower():
                        logger.warning(f"중복 키 발견, 건너뜀: {i+1}~{i+len(batch)}")
                        # 개별 삽입으로 재시도 (중복 건너뛰기)
                        for record in batch:
                            try:
                                session.merge(CardUserInfo(**record))
                                session.commit()
                                inserted_count += 1
                            except Exception as individual_error:
                                session.rollback()
                                logger.debug(f"개별 레코드 삽입 실패: {record.get('발급회원번호', 'Unknown')} - {individual_error}")
                    else:
                        # 중복이 아닌 다른 에러인 경우 다음 배치로 진행
                        continue
        
        logger.info(f"삽입 완료: {inserted_count:,}개 성공, {error_count:,}개 실패")
        
    except Exception as e:
        session.rollback()
        logger.error(f"전체 삽입 프로세스 실패: {e}", exc_info=True)
        raise
    
    finally:
        session.close()
        engine.dispose()
    
    return inserted_count


def insert_card_user_info_safe(
    csv_path: Union[str, Path],
    batch_size: int = 1000,
    create_table: bool = False,
    **csv_kwargs
) -> int:
    """
    안전한 삽입 (중복 키 자동 처리)
    
    Args:
        csv_path: CSV 파일 경로
        batch_size: 배치 크기
        create_table: 테이블 생성 여부
        **csv_kwargs: pandas.read_csv에 전달할 추가 인자
    
    Returns:
        삽입된 레코드 수
    """
    return bulk_insert_card_user_info(
        csv_path=csv_path,
        batch_size=batch_size,
        skip_duplicates=True,
        create_table=create_table,
        **csv_kwargs
    )


if __name__ == "__main__":
    # 사용 예시
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python -m src.postgre.insert <csv_file_path>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        inserted = insert_card_user_info_safe(
            csv_path=csv_file,
            batch_size=1000,
            create_table=True,
            encoding='utf-8'  # 필요에 따라 변경
        )
        print(f"\n총 {inserted:,}개의 레코드가 삽입되었습니다.")
    except Exception as e:
        logger.error(f"삽입 실패: {e}", exc_info=True)
        sys.exit(1)

