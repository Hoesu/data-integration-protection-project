import logging

from sqlalchemy import Engine

from src.database import Base, execute_query
from src.preprocess import allocate_metadata, impute_data

logger = logging.getLogger('project.builder')


class DataProtectionPipeline:

    def __init__(self, config: dict):
        self.config = config

    def _insert_data(self, engine: Engine):
        execute_query(engine, self.config)

    def _prepare_data(self, engine: Engine):
        self.data = execute_query(engine, self.config)
        self.metadata = allocate_metadata(self.data, self.config)
        self.data = impute_data(self.data, self.metadata, self.config)

    def _build_graph(self):
        pass

    def _calculate_risk(self):
        pass

    def _save_results(self):
        pass

    def run(self) -> None:
        logger.info('Pipeline started')
        engine = Base.get_engine()
        Base.metadata.create_all(engine)

        if self.config['data']['action'] == 'insert':
            self._insert_data()
            logger.info(f'Data inserted: {self.inserted_rows} rows')
            logger.info('Pipeline completed (data insertion mode)')
            return

        self._prepare_data()
        logger.info(f'Data shape: {self.data.shape}')
        logger.info(f'Properties: {self.properties}')

        self._build_graph()
        logger.info('Graph built')

        self._calculate_risk()
        logger.info('Risk calculated')

        self._save_results()
        logger.info('Results saved')

        logger.info('Pipeline completed')