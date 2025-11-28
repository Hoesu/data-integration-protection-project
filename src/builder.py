import logging
from pathlib import Path

import networkx as nx
import numpy as np
from sqlalchemy import Engine

from src.database import Base, execute_query
from src.metric import pairwise_distance
from src.preprocess import allocate_metadata, impute_data
from src.utils import visualize_graph

logger = logging.getLogger('project.builder')


class DataProtectionPipeline:

    def __init__(self, config: dict, result_dir: Path):
        self.config = config
        self.result_dir = result_dir

    def _insert_data(self, engine: Engine):
        execute_query(engine, self.config)

    def _prepare_data(self, engine: Engine):
        self.data = execute_query(engine, self.config)
        self.metadata = allocate_metadata(self.data, self.config)
        self.data = impute_data(self.data, self.metadata, self.config)

    def _build_graph(self):
        node_identifier = self.config['graph']['node_identifier']
        scale_parameter = self.config['graph']['scale_parameter']
        threshold = self.config['graph']['edge_threshold']

        distance_matrix = pairwise_distance(
            data=self.data,
            metadata=self.metadata
        )
        adjacency_matrix = np.exp(-distance_matrix / scale_parameter)
        adjacency_matrix[adjacency_matrix < threshold] = 0
        np.fill_diagonal(adjacency_matrix, 0)

        self.graph = nx.from_numpy_array(
            A=adjacency_matrix,
            nodelist=self.data[node_identifier].tolist()
        )

    def _calculate_risk(self):
        pass

    def _save_results(self):
        visualize_graph(
            graph=self.graph,
            result_dir=self.result_dir,
            config=self.config
        )

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

        # self._calculate_risk()
        # logger.info('Risk calculated')

        self._save_results()
        logger.info('Results saved')

        logger.info('Pipeline completed')