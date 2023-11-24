import polars as pl
from typing import Sequence


def pivot_data(
    data: pl.DataFrame,
    columns: list[str],
    values_column: str | Sequence[str],
    index_column: str | Sequence[str],
    feature_column: str | Sequence[str]
) -> pl.DataFrame:
    """
    Pivot the data based on specified columns, values column, index columns, and feature columns.

    Args:
        data (pl.DataFrame): The input DataFrame.
        columns (list[str]): The columns to select from the input data.
        values_column (str | Sequence[str]): The column(s) containing the values to pivot.
        index_column (str | Sequence[str]): The column(s) to use as the index for the pivot table.
        feature_column (str | Sequence[str]): The column(s) to use as the feature(s) for the pivot table.

    Returns:
        pl.DataFrame: The pivot table.
    """
    _data = data.select(columns)
    _data = _data.hstack([pl.Series("Count", [1] * len(_data))])
    _data = _data.pivot(
        values=values_column, index=index_column, columns=feature_column
    ).fill_null(0)
    return _data
