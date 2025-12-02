import logging
from pathlib import Path

import networkx as nx
import numpy as np
from sqlalchemy import Engine

from src.database import Base, execute_query
from src.metric import pairwise_distance
from src.preprocess import allocate_metadata, impute_data
from src.risk import calculate_risk
from src.utils import (
    save_risk_results,
    visualize_adjacency_matrix,
    visualize_graph,
)

logger = logging.getLogger('project.builder')


class DataProtectionPipeline:

    def __init__(self, config: dict, result_dir: Path):
        self.config = config
        self.result_dir = result_dir

    def _insert_data(self, engine: Engine):
        self.inserted_rows = execute_query(engine, self.config)

    def _prepare_data(self, engine: Engine):
        fetch_size = self.config['data']['limit']
        sample_size = self.config['data']['sample_size']
        if sample_size < fetch_size:
            self.data = execute_query(engine, self.config).sample(sample_size)
            self.data.reset_index(drop=True, inplace=True)
        else:
            self.data = execute_query(engine, self.config)
        self.metadata = allocate_metadata(self.data, self.config)
        self.data = impute_data(self.data, self.metadata, self.config)

    def _build_graph(self):
        node_identifier = self.config['graph']['node_identifier']
        threshold = self.config['graph']['edge_threshold']

        distance_matrix = pairwise_distance(
            data=self.data,
            metadata=self.metadata
        )
        adjacency_matrix = 1 - distance_matrix
        adjacency_matrix[adjacency_matrix < threshold] = 0
        np.fill_diagonal(adjacency_matrix, 0)
        self.adjacency_matrix = adjacency_matrix

        self.graph = nx.from_numpy_array(
            A=adjacency_matrix,
            nodelist=self.data[node_identifier].tolist()
        )
        logger.debug(f'Graph nodes: {len(self.graph.nodes())}')
        logger.debug(f'Graph edges: {len(self.graph.edges())}')

    def _calculate_risk(self):
        self.risk_results = calculate_risk(
            graph=self.graph,
            data=self.data,
            config=self.config
        )

    def _save_results(self):
        visualize_adjacency_matrix(
            adjacency_matrix=self.adjacency_matrix,
            result_dir=self.result_dir,
            config=self.config
        )
        visualize_graph(
            graph=self.graph,
            result_dir=self.result_dir,
            config=self.config,
            risk_results=self.risk_results
        )
        save_risk_results(
            risk_results=self.risk_results,
            result_dir=self.result_dir
        )

    def run(self) -> None:
        logger.info('🚀 Pipeline started')
        engine = Base.get_engine()
        Base.metadata.create_all(engine)

        if self.config['data']['action'] == 'insert':
            self._insert_data()
            logger.info(f'Data inserted: {self.inserted_rows} rows')
            logger.info('Pipeline completed (data insertion mode)')
            return

        logger.info('💽 Initiating data preparation')
        self._prepare_data(engine)
        logger.info('💽 Data preparation complete')

        logger.info('🧩 Initiating graph construction')
        self._build_graph()
        logger.info('🧩 Graph construction complete')

        logger.info('⚠️ Initiating risk calculation')
        self._calculate_risk()
        logger.info('⚠️ Risk calculation complete')

        logger.info('💾 Saving results')
        self._save_results()
        logger.info('💾 Results saved')

        logger.info('🎉 Pipeline completed')