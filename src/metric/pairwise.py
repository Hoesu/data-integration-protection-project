import numpy as np
import pandas as pd

from .heom import heom_distance


def pairwise_distance(
    data: pd.DataFrame,
    metadata: dict,
) -> np.ndarray[np.float32]:
    """데이터셋 전체에 대한 HEOM 거리 행렬 계산.

    Parameters
    ----------
    data : pd.DataFrame
        거리를 계산할 데이터셋
    metadata : dict
        컬럼별 메타데이터

    Returns
    -------
    np.ndarray[np.float32]: 거리 행렬
    """
    distance_matrix = np.zeros((len(data), len(data)), dtype=np.float32)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance_matrix[i, j] = heom_distance(
                x=data.iloc[i].to_dict(), y=data.iloc[j].to_dict(), metadata=metadata
            )
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix
