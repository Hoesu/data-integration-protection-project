from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from .tables import *


def _insert_data(
    engine: Engine,
    csv_path: str,
    target_table: Base,
    batch_size: int = 1000,
) -> int:
    """
    데이터 삽입용 함수

    Parameters
    ----------
        engine: sqlalchemy.Engine
            SQLAlchemy 엔진
        csv_path: str
            CSV 파일 절대경로
        target_table: database.tables.Base
            삽입할 테이블
        batch_size: int
            삽입 배치 크기

    Returns
    -------
        int
            삽입된 레코드 수

    Examples
    --------
        >>> inserted_count = insert_data(
        ...     engine=engine,
        ...     csv_path="data/201807_회원정보.csv",
        ...     target_table=CardUserInfo,
        ...     batch_size=1000,
        ... )
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError()

    records = pd.read_csv(Path(csv_path), encoding='utf-8').to_dict('records')

    with Session(engine) as session:
        batch = []
        inserted_count = 0
        try:
            with tqdm(total=len(records), desc='Inserting data', unit='row') as pbar:
                for row in records:
                    batch.append(target_table(**row))
                    if len(batch) >= batch_size:
                        session.add_all(batch)
                        session.flush()
                        inserted_count += len(batch)
                        pbar.update(len(batch))
                        batch.clear()
                if batch:
                    session.add_all(batch)
                    session.flush()
                    inserted_count += len(batch)
                    pbar.update(len(batch))

            session.commit()
        except Exception:
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
    데이터 조회용 함수

    Parameters
    ----------
        engine: sqlalchemy.Engine
            SQLAlchemy 엔진
        source_table: database.tables.Base
            쿼리를 실행할 테이블
        limit: int | None
            쿼리 결과 최대 레코드 수
        filters: dict[str, Any] | None
            필터링 조건 (예: {'컬럼명': 값})
        order_by: list[str] | None
            정렬할 컬럼명 (예: ['컬럼명1', '-컬럼명2'] - 앞에 붙으면 DESC)

    Returns
    -------
        pd.DataFrame
            조회된 데이터

    Examples
    --------
        >>> df = select_data(
        ...     engine=engine,
        ...     source_table=CardUserInfo,
        ...     limit=100,
        ...     filters={'회원여부_이용가능': 1},
        ...     order_by=['발급회원번호']
        ... )
    """
    with Session(engine) as session:
        try:
            query = select(source_table)

            if filters:
                for column_name, value in filters.items():
                    if hasattr(source_table, column_name):
                        column = getattr(source_table, column_name)

                        if isinstance(value, (list, tuple)):
                            query = query.where(column.in_(value))
                        elif value is None:
                            query = query.where(column.is_(None))
                        else:
                            query = query.where(column == value)
                    else:
                        raise ValueError()

            if order_by:
                for order_col in order_by:
                    if order_col.startswith('-'):
                        col_name = order_col[1:]
                        if hasattr(source_table, col_name):
                            query = query.order_by(
                                getattr(source_table, col_name).desc()
                                )
                        else:
                            raise ValueError()
                    else:
                        if hasattr(source_table, order_col):
                            query = query.order_by(
                                getattr(source_table, order_col).asc()
                                )
                        else:
                            raise ValueError()

            if limit is not None:
                query = query.limit(limit)

            result = session.execute(query)
            rows = result.scalars().all()

            if not rows:
                return pd.DataFrame()

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
    table_map = {
        "CardUserInfo": CardUserInfo,
        "CardCreditInfo": CardCreditInfo,
        "IndividualCB": IndividualCB,
    }
    if table_name not in table_map:
        raise ValueError()
    return table_map[table_name]


def execute_query(engine: Engine, config: dict) -> int | pd.DataFrame:
    """
    """
    if config["action"] == "insert":
        return _insert_data(
            engine=engine,
            csv_path=Path(config["insert"]["csv_path"]),
            target_table=_get_table_class(config["insert"]["target_table"]),
            batch_size=config["insert"]["batch_size"]
        )
    elif config["action"] == "select":
        return _select_data(
            engine=engine,
            source_table=_get_table_class(config["select"]["source_table"]),
            limit=config["select"]["limit"],
            filters=config["select"]["filters"],
            order_by=config["select"]["order_by"],
        )
    else:
        raise ValueError()