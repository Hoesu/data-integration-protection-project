import os
from datetime import date
from typing import Any

import dotenv
import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase

dotenv.load_dotenv()


class Base(DeclarativeBase):
    @staticmethod
    def get_engine() -> Engine:
        engine = create_engine(
            url='postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'.format(
                user=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWORD'),
                host=os.getenv('POSTGRES_HOST'),
                port=os.getenv('POSTGRES_PORT'),
                database=os.getenv('POSTGRES_DB'),
            ),
            echo=True,
        )
        return engine

    @staticmethod
    def _normalize_missing_value(value: Any) -> Any:
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, str) and value.strip().lower() == 'nan':
            return None
        return value

    @staticmethod
    def _normalize_yyyymmdd_date(value: int | float) -> date:
        try:
            return date(int(value // 10000), int((value % 10000) // 100), int(value % 100))
        except ValueError:
            return None

    @staticmethod
    def _normalize_yyyymm_date(value: int | float) -> date:
        try:
            return date(int(value // 100), int(value % 100), 1)
        except ValueError:
            return None
