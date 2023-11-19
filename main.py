import polars as pl

from algorithms.association import apriori, summary
from config import BLACK_FRIDAY_DATASET_PATH
from utils.process import pivot_data

if __name__ == "__main__":
    data = pl.read_csv(BLACK_FRIDAY_DATASET_PATH)
    sale_data = pivot_data(
        data, ["User_ID", "Product_ID"], "Count", "User_ID", "Product_ID"
    )
    columns = apriori(sale_data, drop_columns=["User_ID"])
    result = summary(sale_data, columns)
    print(result)
    result.write_csv("result.csv")
    pass
