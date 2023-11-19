import polars as pl

from config import APRIORI_MIN_CONFIDENCE, APRIORI_MIN_SUPPORT


def summary(
    data: pl.DataFrame,
    columns: list[list[str]],
) -> pl.DataFrame:
    schema = [
        ("Antecedents", str),
        ("Consequents", str),
        ("Antecedent Support", pl.Float64),
        ("Consequent Support", pl.Float64),
        ("Support", pl.Float64),
        ("Confidence", pl.Float64),
        ("Lift", pl.Float64),
        ("Leverage", pl.Float64),
    ]
    _shape = data.shape[0]
    result = pl.DataFrame(schema=schema)
    for col1, col2 in columns:
        support = (data[col1] * data[col2]).sum() / _shape
        confidence = support * _shape / data[col1].sum()

        col1_sum = data[col1].sum()
        col2_sum = data[col2].sum()
        _ = pl.DataFrame(
            [
                [
                    col1,
                    col2,
                    col1_sum / _shape,
                    col2_sum / _shape,
                    support,
                    confidence,
                    confidence * _shape / col2_sum,
                    support - (col1_sum / _shape) * (col2_sum / _shape),
                ]
            ],
            schema=schema,
        )
        result = result.vstack(_)
    return result


def apriori(
    data: pl.DataFrame,
    drop_columns: list[str],
    min_support: float = APRIORI_MIN_SUPPORT,
    min_confidence: float = APRIORI_MIN_CONFIDENCE,
) -> list[list[str]]:
    """
    :param data: pl.DataFrame
    :param drop_columns: list[str]
    :param min_support: float
    :param min_confidence: float
    :return: list[list[str]]
    """

    col_list = []
    _data = data.drop(drop_columns)
    _shape = _data.shape[0]
    for col in _data.columns:
        if _data[col].sum() / _shape < min_support:
            _data = _data.drop(col)

    for col1 in _data.columns:
        for col2 in _data.columns:
            gross_support = (_data[col1] * _data[col2]).sum()
            if (
                col1 != col2
                and gross_support / _shape > min_support
                and gross_support * _shape / _data[col1].sum() > min_confidence
            ):
                col_list.append([col1, col2])

    return col_list


def frequent_pattern_growth(data: pl.DataFrame) -> pl.DataFrame:
    pass
