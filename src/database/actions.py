from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from .tables import Base, CardCreditInfo, CardUserInfo, IndividualCB


def _insert_data(
    engine: Engine,
    csv_path: str,
    target_table: Base,
    batch_size: int = 1000,
) -> int:
    """
    CSV 파일에서 데이터를 읽어 데이터베이스 테이블에 배치로 삽입합니다.

    CSV 파일을 읽어 SQLAlchemy ORM 객체로 변환한 후, 지정된 배치 크기만큼
    묶어서 데이터베이스에 삽입합니다. 진행 상황은 tqdm을 통해 표시됩니다.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy 엔진 객체.
    csv_path : str
        삽입할 데이터가 포함된 CSV 파일의 경로.
    target_table : Base
        데이터를 삽입할 대상 테이블 클래스 (SQLAlchemy ORM 모델).
    batch_size : int, optional
        한 번에 삽입할 레코드 수. 기본값은 1000입니다.

    Returns
    -------
    int
        성공적으로 삽입된 레코드의 총 개수.

    Raises
    ------
    FileNotFoundError
        지정된 CSV 파일이 존재하지 않을 때 발생합니다.
    Exception
        데이터 삽입 중 오류가 발생하면 롤백 후 예외를 재발생시킵니다.

    Notes
    -----
    - CSV 파일은 UTF-8 인코딩을 사용해야 합니다.
    - 배치 삽입을 통해 대용량 데이터 처리 시 메모리 효율성을 높입니다.
    - 트랜잭션 실패 시 자동으로 롤백됩니다.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError()

    ## CSV 파일을 읽어 딕셔너리 리스트로 변환
    records = pd.read_csv(Path(csv_path), encoding='utf-8').to_dict('records')

    with Session(engine) as session:
        batch = []
        inserted_count = 0
        try:
            ## 진행 상황 표시를 위한 tqdm 진행바 초기화
            with tqdm(total=len(records), desc='Inserting data', unit='row') as pbar:
                for row in records:
                    ## 각 행을 ORM 객체로 변환하여 배치에 추가
                    batch.append(target_table(**row))
                    ## 배치 크기에 도달하면 데이터베이스에 삽입
                    if len(batch) >= batch_size:
                        session.add_all(batch)
                        session.flush()
                        inserted_count += len(batch)
                        pbar.update(len(batch))
                        batch.clear()
                ## 남은 레코드가 있으면 마지막 배치로 삽입
                if batch:
                    session.add_all(batch)
                    session.flush()
                    inserted_count += len(batch)
                    pbar.update(len(batch))
            ## 모든 배치 삽입 완료 후 커밋
            session.commit()
        except Exception:
            ## 오류 발생 시 롤백
            session.rollback()
            raise
        finally:
            session.close()
            engine.dispose()

    return inserted_count


def _select_data(
    engine: Engine,
    source_table: Base,
    limit: int | None = None,
    filters: dict[str, Any] | None = None,
    order_by: list[str] | None = None,
) -> pd.DataFrame:
    """
    데이터베이스 테이블에서 데이터를 조회하여 pandas DataFrame으로 반환합니다.

    필터링, 정렬, 제한 등의 조건을 적용하여 데이터를 조회할 수 있습니다.
    조회된 데이터는 pandas DataFrame 형태로 반환됩니다.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy 엔진 객체.
    source_table : Base
        조회할 소스 테이블 클래스 (SQLAlchemy ORM 모델).
    limit : int | None, optional
        반환할 최대 레코드 수. None이면 모든 레코드를 반환합니다.
        기본값은 None입니다.
    filters : dict[str, Any] | None, optional
        필터링 조건 딕셔너리. 키는 컬럼명, 값은 필터링할 값입니다.
        값이 리스트나 튜플이면 IN 조건으로 처리됩니다.
        None이면 IS NULL 조건으로 처리됩니다.
        기본값은 None입니다.
    order_by : list[str] | None, optional
        정렬할 컬럼명 리스트. 컬럼명 앞에 '-'가 붙으면 내림차순,
        없으면 오름차순으로 정렬합니다.
        기본값은 None입니다.

    Returns
    -------
    pd.DataFrame
        조회된 데이터를 담은 pandas DataFrame. 결과가 없으면 빈 DataFrame을 반환합니다.

    Raises
    ------
    ValueError
        filters나 order_by에 지정된 컬럼명이 테이블에 존재하지 않을 때 발생합니다.
    Exception
        쿼리 실행 중 오류가 발생하면 예외를 재발생시킵니다.

    Examples
    --------
    >>> filters = {'user_id': [1, 2, 3], 'status': 'active'}
    >>> order_by = ['-created_at', 'user_id']
    >>> df = _select_data(engine, CardUserInfo, limit=100, filters=filters, order_by=order_by)
    """
    with Session(engine) as session:
        try:
            ## 기본 SELECT 쿼리 생성
            query = select(source_table)
            ## 필터 조건 적용
            if filters:
                for column_name, value in filters.items():
                    if hasattr(source_table, column_name):
                        column = getattr(source_table, column_name)
                        ## 리스트/튜플인 경우 IN 조건으로 처리
                        if isinstance(value, (list, tuple)):
                            query = query.where(column.in_(value))
                        ## None인 경우 IS NULL 조건으로 처리
                        elif value is None:
                            query = query.where(column.is_(None))
                        ## 그 외의 경우 등호 조건으로 처리
                        else:
                            query = query.where(column == value)
                    else:
                        raise ValueError()
            ## 정렬 조건 적용
            if order_by:
                for order_col in order_by:
                    ## '-'로 시작하면 내림차순 정렬
                    if order_col.startswith('-'):
                        col_name = order_col[1:]
                        if hasattr(source_table, col_name):
                            query = query.order_by(
                                getattr(source_table, col_name).desc()
                            )
                        else:
                            raise ValueError()
                    ## 그 외의 경우 오름차순 정렬
                    else:
                        if hasattr(source_table, order_col):
                            query = query.order_by(
                                getattr(source_table, order_col).asc()
                            )
                        else:
                            raise ValueError()
            ## 레코드 수 제한 적용
            if limit is not None:
                query = query.limit(limit)
            ## 쿼리 실행 및 결과 가져오기
            result = session.execute(query)
            rows = result.scalars().all()
            ## 결과가 없으면 빈 DataFrame 반환
            if not rows:
                return pd.DataFrame()
            ## ORM 객체를 딕셔너리로 변환하여 리스트에 추가
            results = []
            for row in rows:
                row_dict = {
                    column.name: getattr(row, column.name)
                    for column in source_table.__table__.columns
                }
                results.append(row_dict)
            return pd.DataFrame(results)

        except Exception:
            raise

        finally:
            session.close()
            engine.dispose()


def _get_table_class(table_name: str):
    """
    테이블 이름에 해당하는 SQLAlchemy ORM 클래스를 반환합니다.

    문자열로 된 테이블 이름을 받아 해당하는 테이블 클래스를 반환하는
    헬퍼 함수입니다.

    Parameters
    ----------
    table_name : str
        조회할 테이블 이름. 지원되는 값: 'CardUserInfo', 'CardCreditInfo', 'IndividualCB'.

    Returns
    -------
    Base
        해당 테이블 이름에 매핑된 SQLAlchemy ORM 클래스.

    Raises
    ------
    ValueError
        지원되지 않는 테이블 이름이 전달될 때 발생합니다.

    Examples
    --------
    >>> table_class = _get_table_class('CardUserInfo')
    >>> isinstance(table_class, type)
    True
    """
    ## 테이블 이름과 클래스 매핑 딕셔너리
    table_map = {
        'CardUserInfo': CardUserInfo,
        'CardCreditInfo': CardCreditInfo,
        'IndividualCB': IndividualCB,
    }
    if table_name not in table_map:
        raise ValueError()
    return table_map[table_name]


def execute_query(engine: Engine, config: dict) -> int | pd.DataFrame:
    """
    설정 딕셔너리에 따라 데이터 삽입 또는 조회 쿼리를 실행합니다.

    config 딕셔너리의 'action' 값에 따라 데이터 삽입 또는 조회 작업을 수행합니다.
    'insert'인 경우 CSV 파일에서 데이터를 읽어 삽입하고,
    그 외의 경우 데이터를 조회하여 반환합니다.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy 엔진 객체.
    config : dict
        쿼리 실행 설정을 담은 딕셔너리. 다음 키를 포함해야 합니다:
        - 'data': 중첩 딕셔너리
            - 'action': str, 'insert' 또는 'select'
            - insert의 경우:
                - 'csv_path': str, CSV 파일 경로
                - 'target_table': str, 대상 테이블 이름
                - 'batch_size': int, 배치 크기
            - select의 경우:
                - 'source_table': str, 소스 테이블 이름
                - 'limit': int | None, 레코드 수 제한
                - 'filters': dict[str, Any] | None, 필터 조건
                - 'order_by': list[str] | None, 정렬 조건

    Returns
    -------
    int | pd.DataFrame
        action이 'insert'인 경우 삽입된 레코드 수(int)를 반환하고,
        그 외의 경우 조회된 데이터를 담은 pandas DataFrame을 반환합니다.

    Raises
    ------
    FileNotFoundError
        insert 작업 시 지정된 CSV 파일이 존재하지 않을 때 발생합니다.
    ValueError
        지원되지 않는 테이블 이름이 전달되거나 필터/정렬 조건에 잘못된 컬럼명이 포함될 때 발생합니다.
    Exception
        쿼리 실행 중 오류가 발생하면 예외를 재발생시킵니다.

    Examples
    --------
    >>> # 데이터 삽입 예제
    >>> insert_config = {
    ...     'data': {
    ...         'action': 'insert',
    ...         'csv_path': '/path/to/data.csv',
    ...         'target_table': 'CardUserInfo',
    ...         'batch_size': 1000
    ...     }
    ... }
    >>> count = execute_query(engine, insert_config)
    >>>
    >>> # 데이터 조회 예제
    >>> select_config = {
    ...     'data': {
    ...         'action': 'select',
    ...         'source_table': 'CardUserInfo',
    ...         'limit': 100,
    ...         'filters': {'status': 'active'},
    ...         'order_by': ['-created_at']
    ...     }
    ... }
    >>> df = execute_query(engine, select_config)
    """
    ## 삽입 작업인 경우
    if config['data']["action"] == 'insert':
        return _insert_data(
            engine=engine,
            csv_path=config['data']['csv_path'],
            target_table=_get_table_class(config['data']['target_table']),
            batch_size=config['data']['batch_size'],
        )
    ## 조회 작업인 경우
    return _select_data(
        engine=engine,
        source_table=_get_table_class(config['data']['source_table']),
        limit=config['data']['limit'],
        filters=config['data']['filters'],
        order_by=config['data']['order_by']
    )
