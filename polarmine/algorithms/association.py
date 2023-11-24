from abc import abstractmethod
from typing import Type

import polars as pl
from tqdm import tqdm

from config import ASSOCIATION_MIN_SUPPORT, ASSOCIATION_MIN_VALUE, ASSOCIATION_RULE, ITEMSET_LENGTH


class BaseAssociationSummary:
    """
    Base abstract class for summary

    Parameters:
        data (pl.DataFrame, optional): The input data. Defaults to None.
        frequent_itemsets (list, optional): The frequent itemsets. Defaults to None.
        itemset_length (int, optional): The length of itemsets. Defaults to ITEMSET_LENGTH.

    Attributes:
        data (pl.DataFrame): The input data.
        length (int): The number of rows in the data.
        frequent_itemsets (list): The frequent itemsets.
        itemset_length (int): The length of itemsets.

    Methods:
        summary(): The abstract method to be implemented by subclasses, which returns a summary.
    """

    def __init__(self,
                 data: pl.DataFrame = None,
                 frequent_itemsets: list = None,
                 itemset_length: int = ITEMSET_LENGTH):
        """Initialize the class with data, frequent_itemsets, and itemset_length."""
        self.data = data
        self.length = self.data.shape[0]
        self.frequent_itemsets = frequent_itemsets
        self.itemset_length = itemset_length

    @abstractmethod
    def summary(self) -> pl.DataFrame:
        pass


class FrequentItemsetsSummary(BaseAssociationSummary):
    """Summary Frequent Itemsets

    Parameters:
        data (pl.DataFrame, optional): The input data. Defaults to None.
        frequent_itemsets (list, optional): The frequent itemsets. Defaults to None.
        itemset_length (int, optional): The length of itemsets. Defaults to ITEMSET_LENGTH.

    Attributes:
        data (pl.DataFrame): The input data.
        length (int): The number of rows in the data.
        frequent_itemsets (list): The frequent itemsets.
        itemset_length (int): The length of itemsets.

    Methods:
        summary(): Returns a summary of the frequent itemsets.

    Examples:
        ```python
        import polars as pl
        from algorithms.association import FrequentItemsetsSummary
        from utils.process import pivot_data
        data = pl.read_csv("./src/BlackFridaytrain.csv")
        sale_data = pivot_data(data, ["User_ID", "Product_ID"], "Count", "User_ID", "Product_ID").drop(["User_ID"])
        apriori = Apriori(sale_data)
        apriori.fit()
        frequent_itemsets = apriori.frequent_itemsets
        Summary = FrequentItemsetsSummary(apriori.data, frequent_itemsets, 2)
        Summary.summary()
        ```
    """

    def __init__(self,
                 data: pl.DataFrame = None,
                 frequent_itemsets: list = None,
                 itemset_length: int = ITEMSET_LENGTH, ):
        super().__init__(data, frequent_itemsets, itemset_length)

    def summary(self) -> pl.DataFrame:
        if self.frequent_itemsets is None:
            raise ValueError("Frequent itemsets not found")
        # Create a new DataFrame to store the _summary
        _summary = pl.DataFrame(schema=[("Length", int), ("Itemsets", pl.Object), ("Support", pl.Float64)])
        # Iterate over each itemset
        for itemset in self.frequent_itemsets:
            # Calculate the support for each item in the itemset
            support = pl.Series([1] * self.length)
            for item in itemset:
                support *= self.data[item]
            # Add a new row to the _summary DataFrame
            new_row = pl.DataFrame(
                [{"Length": self.itemset_length, "Itemsets": itemset, "Support": support.sum() / self.length}])
            _summary = _summary.vstack(new_row)
        return _summary


class AssociationRulesSummary(BaseAssociationSummary):
    """Summary Association Rules"""
    def __init__(self,
                 data: pl.DataFrame = None,
                 frequent_itemsets: list[set[str] | str] | None = None,
                 itemset_length: int = ITEMSET_LENGTH,
                 rule: str = ASSOCIATION_RULE,
                 min_value: float = ASSOCIATION_MIN_VALUE[ASSOCIATION_RULE], ):
        """
        Initialize the class with the given parameters.

        Args:
            data: DataFrame containing the transaction data.
            frequent_itemsets: List of frequent itemsets.
            itemset_length: Length of the itemsets.
            rule: Association rule type to filter the frequent itemsets, default is 'confidence', options are
                'confidence', 'support', 'lift', 'leverage', 'antecedent_support' and 'consequent_support'.
            min_value: Minimum value for the association rule.
        """
        super().__init__(data, frequent_itemsets, itemset_length)
        self.rule = rule
        self.min_value = min_value
        self.schema = [
            ("Antecedents", str),
            ("Consequents", str),
            ("Antecedent Support", pl.Float64),
            ("Consequent Support", pl.Float64),
            ("Support", pl.Float64),
            ("Confidence", pl.Float64),
            ("Lift", pl.Float64),
            ("Leverage", pl.Float64)
        ]

    def summary(self) -> pl.DataFrame:
        """
        Generate a summary of frequent itemsets and association rules.

        Returns:
            result (pl.DataFrame): A DataFrame containing the summary.
        """
        # Check if frequent itemsets of length 2 exist
        if self.itemset_length != 2:
            raise AssertionError("Frequent itemsets of length 2 not found")

        # Create an empty DataFrame with the specified schema
        result = pl.DataFrame(schema=self.schema)

        # Iterate over each pair of frequent itemsets
        for col1, col2 in self.frequent_itemsets:
            # Calculate association rules for col1 and col2
            _ = self.calculate_association_rules(col1, col2, self.rule, self.min_value)
            # If the association rules are not None, append them to the result DataFrame
            if _ is not None:
                result = result.vstack(_)

            # Reverse the order of col1 and col2
            _ = self.calculate_association_rules(col2, col1, self.rule, self.min_value)
            if _ is not None:
                result = result.vstack(_)

        return result

    def calculate_association_rules(self, antecedents: str, consequents: str, rule: str,
                                    min_value: float) -> pl.DataFrame | None:
        """
        Calculate the association rules based on the given parameters, rule and minimum value condition.

        Args:
            antecedents: Name of the antecedents column in the data.
            consequents: Name of the consequents column in the data.
            rule: Association rule type to filter the rules.
            min_value: Minimum value for the association rule.

        Returns:
            result (pl.DataFrame) if the rule meets the minimum value condition, otherwise None.
        """
        # Get the antecedents and consequents columns from the data
        _antecedents = self.data[antecedents]
        _consequents = self.data[consequents]
        # Calculate metrics
        antecedent_support = _antecedents.sum() / self.length
        consequent_support = _consequents.sum() / self.length
        support = (_antecedents * _consequents).sum() / self.length
        confidence = support / antecedent_support
        lift = support / (antecedent_support * consequent_support)
        leverage = support - antecedent_support * consequent_support
        # Create a DataFrame with the calculated metrics
        result = pl.DataFrame(
            [
                [
                    antecedents,
                    consequents,
                    antecedent_support,
                    consequent_support,
                    support,
                    confidence,
                    lift,
                    leverage,
                ]
            ],
            schema=self.schema,
        )
        # Check the rule type and return the result if it meets the minimum value condition
        match rule:
            case "antecedent_support":
                if antecedent_support >= min_value:
                    return result
            case "consequent_support":
                if consequent_support >= min_value:
                    return result
            case "support":
                if support >= min_value:
                    return result
            case "confidence":
                if confidence >= min_value:
                    return result
            case "lift":
                if lift >= min_value:
                    return result
            case "leverage":
                if leverage >= min_value:
                    return result
            case _:
                raise ValueError(f"Invalid rule: {rule}")
        # If the rule type is not recognized or the minimum value condition is not met, return None
        return None


class BaseAssociation:
    """Base abstract class for association algorithms"""

    def __init__(self, data: pl.DataFrame = None, itemset_length=ITEMSET_LENGTH, min_support=ASSOCIATION_MIN_SUPPORT):
        """
        Initialize the BaseAssociation class with the given parameters.

        Args:
            data: DataFrame containing the transaction data.
            itemset_length: Length of the itemsets.
            min_support: Minimum support for frequent itemsets.
        """
        self.data = data
        self.length = data.shape[0]
        self.frequent_itemsets = list()
        self.itemset_length = itemset_length
        self.min_support = min_support

    @abstractmethod
    def fit(self, itemset_length=ITEMSET_LENGTH, min_support=ASSOCIATION_MIN_SUPPORT):
        pass


class Apriori(BaseAssociation):
    def __init__(self, data: pl.DataFrame, itemset_length: int = ITEMSET_LENGTH, min_support=ASSOCIATION_MIN_SUPPORT):
        super().__init__(data, itemset_length, min_support)
        self.current_itemset_length = 1

    def fit(self, min_support=ASSOCIATION_MIN_SUPPORT, itemset_length=ITEMSET_LENGTH) -> list:
        self.itemset_length, self.min_support = itemset_length, min_support
        self.frequent_itemsets = self.generate_frequent_one_itemsets

        while self.current_itemset_length < self.itemset_length and self.frequent_itemsets:
            self.current_itemset_length += 1
            self.frequent_itemsets = self.generate_frequent_k_itemsets
        return self.frequent_itemsets

    @property
    def generate_frequent_one_itemsets(self) -> list[str | None]:
        frequent_one_itemsets = list()
        for col in self.data.columns:
            if self.data[col].sum() / self.length >= self.min_support:
                frequent_one_itemsets.append(col)
        return frequent_one_itemsets

    @property
    def generate_frequent_k_itemsets(self) -> list[set[str] | None]:
        candidates = self.generate_candidates(self.frequent_itemsets, self.current_itemset_length)
        frequent_k_itemsets = list()
        for candidate in tqdm(candidates):
            support = pl.Series([1] * self.length)
            for item in candidate:
                support *= self.data[item]
            if support.sum() / self.length >= self.min_support:
                frequent_k_itemsets.append(candidate)
        return frequent_k_itemsets

    @staticmethod
    def generate_candidates(prev_itemsets, k) -> list:
        candidates = list()
        unique_candidates = set()

        for i, itemset1 in tqdm(enumerate(prev_itemsets)):
            for itemset2 in prev_itemsets[i + 1:]:
                if k - 1 == 1:
                    union_set = {itemset1, }.union({itemset2, })
                else:
                    union_set = set(itemset1).union(set(itemset2))
                if len(union_set) == k:
                    tuple_union_set = tuple(sorted(union_set))
                    if tuple_union_set not in unique_candidates:
                        unique_candidates.add(tuple_union_set)
                        candidates.append(set(tuple_union_set))

        return candidates


class FrequentPatternGrowth(BaseAssociation):
    def fit(self, itemset_length=ITEMSET_LENGTH, min_support=ASSOCIATION_MIN_SUPPORT):
        pass

    def summary(self, summary_type: Type[BaseAssociationSummary] = BaseAssociationSummary):
        pass
