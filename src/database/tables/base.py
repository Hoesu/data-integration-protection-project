import os
from datetime import date
from typing import Any, Optional

import dotenv
from sqlalchemy import Date, TypeDecorator, create_engine
from sqlalchemy.orm import DeclarativeBase

dotenv.load_dotenv()


class Base(DeclarativeBase):
    pass


class YYYYMMDDDate(TypeDecorator):
    """YYYYMMDD 형식의 정수를 DATE 타입으로 변환하는 커스텀 타입"""
    impl = Date
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        # 정수 또는 float를 문자열로 변환하여 처리
        if isinstance(value, (int, float)):
            value_str = str(int(value))  # float도 정수로 변환
            if len(value_str) == 8:  # YYYYMMDD
                return date(int(value_str[:4]), int(value_str[4:6]), int(value_str[6:8]))
            elif len(value_str) == 6:  # YYYYMM (해당 월의 첫 번째 날로 변환)
                return date(int(value_str[:4]), int(value_str[4:6]), 1)
            else:
                raise ValueError(f"Invalid date format: {value} (expected YYYYMMDD or YYYYMM)")
        # 문자열도 지원 (하위 호환성)
        if isinstance(value, str):
            value = value.strip()
            if not value or value == "_":
                return None
            if len(value) == 8:  # YYYYMMDD
                return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
            elif len(value) == 6:  # YYYYMM
                return date(int(value[:4]), int(value[4:6]), 1)
            else:
                raise ValueError(f"Invalid date format: {value} (expected YYYYMMDD or YYYYMM)")
        raise TypeError(f"Unsupported type: {type(value)}")

    def process_result_value(self, value: Any, dialect: Any) -> Optional[date]:
        return value


class YYYYMMDate(TypeDecorator):
    """YYYYMM 형식의 정수를 DATE 타입으로 변환하는 커스텀 타입 (해당 월의 첫 번째 날)"""
    impl = Date
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        # 정수 또는 float를 문자열로 변환하여 처리
        if isinstance(value, (int, float)):
            value_str = str(int(value))  # float도 정수로 변환
            if len(value_str) == 6:  # YYYYMM
                return date(int(value_str[:4]), int(value_str[4:6]), 1)
            elif len(value_str) == 8:  # YYYYMMDD (일자 무시하고 월의 첫 번째 날로 변환)
                return date(int(value_str[:4]), int(value_str[4:6]), 1)
            else:
                raise ValueError(f"Invalid date format: {value} (expected YYYYMM or YYYYMMDD)")
        # 문자열도 지원 (하위 호환성)
        if isinstance(value, str):
            value = value.strip()
            if not value or value == "_":
                return None
            if len(value) == 6:  # YYYYMM
                return date(int(value[:4]), int(value[4:6]), 1)
            elif len(value) == 8:  # YYYYMMDD (일자 무시하고 월의 첫 번째 날로 변환)
                return date(int(value[:4]), int(value[4:6]), 1)
            else:
                raise ValueError(f"Invalid date format: {value} (expected YYYYMM or YYYYMMDD)")
        raise TypeError(f"Unsupported type: {type(value)}")

    def process_result_value(self, value: Any, dialect: Any) -> Optional[date]:
        return value


def get_engine():
    engine = create_engine(
        url="postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            database=os.getenv("POSTGRES_DB"),
        ),
        echo=True
    )
    return engine
