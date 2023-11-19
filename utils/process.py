import polars as pl


def pivot_data(
    data: pl.DataFrame, columns: list[str], values_column, index_column, feature_column
) -> pl.DataFrame:
    _data = data.select(columns)
    _data = _data.hstack([pl.Series("Count", [1] * len(_data))])
    _data = _data.pivot(
        values=values_column, index=index_column, columns=feature_column
    ).fill_null(0)
    return _data
