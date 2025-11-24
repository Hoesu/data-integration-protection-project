from pathlib import Path

import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from tqdm import tqdm

from .tables import Base


def insert_data(
    engine: Engine,
    csv_path: Path,
    target_table: Base,
    batch_size: int = 1000,
) -> int:
    if not Path(csv_path).exists():
        raise FileNotFoundError()

    records = pd.read_csv(
        filepath_or_buffer=Path(csv_path),
        encoding='utf-8',
    ).to_dict('records')

    with Session(engine) as session:
        batch = []
        inserted_count = 0

        try:
            with tqdm(
                total=len(records),
                desc='Inserting data',
                unit='row',
            ) as pbar:

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
