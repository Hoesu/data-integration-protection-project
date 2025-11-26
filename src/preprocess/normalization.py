import pandas as pd

def normalize_data(
    data: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    데이터를 정규화합니다.
    """
    return data