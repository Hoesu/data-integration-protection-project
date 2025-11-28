import logging

import numpy as np
import networkx as nx
from sqlalchemy import Engine

from src.database import Base, execute_query
from src.preprocess import allocate_metadata, impute_data
from src.metric import pairwise_distance

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
        node_identifier = self.config['graph']['node_identifier']
        scale_parameter = self.config['graph']['scale_parameter']
        distance_matrix = pairwise_distance(
            data=self.data,
            metadata=self.metadata
        )
        adjacency_matrix = np.exp(-distance_matrix / scale_parameter)
        self.graph = nx.from_numpy_array(
            A=adjacency_matrix,
            nodelist=self.data[node_identifier].tolist()
        )

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

        self._prepare_data(engine)
        logger.info(f'Data shape: {self.data.shape}')
        logger.info(f'Metadata: {self.metadata}')

        self._build_graph()
        logger.info('Graph built')
        breakpoint()

        self._calculate_risk()
        logger.info('Risk calculated')

        self._save_results()
        logger.info('Results saved')

        logger.info('Pipeline completed')