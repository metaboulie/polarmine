from abc import abstractmethod
from typing import Type

import polars as pl
from tqdm import tqdm

from config import ASSOCIATION_MIN_SUPPORT, ASSOCIATION_MIN_VALUE, ASSOCIATION_RULE, ITEMSET_LENGTH


class BaseAssociationSummary:
    """Base abstract class for _summary"""

    def __init__(self,
                 data: pl.DataFrame = None,
                 frequent_itemsets: list[set[str] | str] | None = None,
                 itemset_length: int = ITEMSET_LENGTH, ):
        """
        Initialize the class with data, frequent_itemsets, and itemset_length.

        Parameters:
            data (pl.DataFrame, optional): The input data. Defaults to None.
            frequent_itemsets (list[set[str] | str] | None, optional): The frequent itemsets. Defaults to None.
            itemset_length (int, optional): The length of itemsets. Defaults to ITEMSET_LENGTH.
        """
        self.data = data
        self.length = self.data.shape[0]
        self.frequent_itemsets = frequent_itemsets
        self.itemset_length = itemset_length

    @abstractmethod
    def summary(self) -> pl.DataFrame:
        pass


class FrequentItemsetsSummary(BaseAssociationSummary):
    """Summary Frequent Itemsets"""

    def __init__(self,
                 data: pl.DataFrame = None,
                 frequent_itemsets: list[set[str] | str] | None = None,
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
    def __init__(self,
                 data: pl.DataFrame = None,
                 frequent_itemsets: list[set[str] | str] | None = None,
                 itemset_length: int = ITEMSET_LENGTH,
                 rule: str = ASSOCIATION_RULE,
                 min_value: float = ASSOCIATION_MIN_VALUE[ASSOCIATION_RULE], ):
        super().__init__(data, frequent_itemsets, itemset_length)
        self.rule = rule
        self.min_value = min_value
        self.schema = [("Antecedents", str),
                       ("Consequents", str),
                       ("Antecedent Support", pl.Float64),
                       ("Consequent Support", pl.Float64),
                       ("Support", pl.Float64),
                       ("Confidence", pl.Float64),
                       ("Lift", pl.Float64),
                       ("Leverage", pl.Float64)]

    def summary(self) -> pl.DataFrame:
        try:
            assert self.itemset_length == 2
        except AssertionError:
            raise AssertionError("Frequent itemsets of length 2 not found")
        result = pl.DataFrame(schema=self.schema)
        for col1, col2 in self.frequent_itemsets:
            _ = self.calculate_association_rules(col1, col2, self.rule, self.min_value)
            result = result.vstack(_)
            _ = self.calculate_association_rules(col2, col1, self.rule, self.min_value)
            result = result.vstack(_)
        return result

    def calculate_association_rules(self, antecedents: str, consequents: str, rule: str,
                                    min_value: float) -> pl.DataFrame:
        _antecedents = self.data[antecedents]
        _consequents = self.data[consequents]
        antecedent_support = _antecedents.sum() / self.length
        consequent_support = _consequents.sum() / self.length
        support = (_antecedents * _consequents).sum() / self.length
        confidence = support / antecedent_support
        lift = support / (antecedent_support * consequent_support)
        leverage = support - antecedent_support * consequent_support
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


class BaseAssociation:
    """Base abstract class for association algorithms"""

    def __init__(self, data: pl.DataFrame = None, itemset_length=ITEMSET_LENGTH, min_support=ASSOCIATION_MIN_SUPPORT):
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
        self.frequent_itemsets_candidates = list()
        self.current_itemset_length = 1

    def fit(self, min_support=ASSOCIATION_MIN_SUPPORT, itemset_length=ITEMSET_LENGTH) -> list:
        self.itemset_length, self.min_support = itemset_length, min_support
        self.frequent_itemsets, self.frequent_itemsets_candidates = self.generate_frequent_one_itemsets

        while self.current_itemset_length < self.itemset_length and self.frequent_itemsets_candidates:
            self.current_itemset_length += 1
            self.frequent_itemsets, self.frequent_itemsets_candidates = self.generate_frequent_k_itemsets

        return self.frequent_itemsets

    @property
    def generate_frequent_one_itemsets(self) -> tuple[list[str | None], list[str | None]]:
        frequent_one_itemsets, frequent_one_itemsets_candidates = list(), list()
        max_support = float((self.data.sum() / self.length).transpose().max().to_numpy())
        for col in self.data.columns:
            if self.data[col].sum() / self.length >= self.min_support:
                frequent_one_itemsets.append(col)
            if self.data[col].sum() / self.length >= self.min_support / max_support:
                frequent_one_itemsets_candidates.append(col)
        return frequent_one_itemsets, frequent_one_itemsets_candidates

    @property
    def generate_frequent_k_itemsets(self) -> tuple[list[set[str] | None], list[set[str] | None]]:
        print(f"length: {self.current_itemset_length}, candidates: {self.frequent_itemsets_candidates}")
        candidates = self.generate_candidates(self.frequent_itemsets_candidates, self.current_itemset_length)
        frequent_k_itemsets = list()
        frequent_k_itemsets_candidates = list()
        agg_support_list = list()
        for candidate in tqdm(candidates):
            support = pl.Series([1] * self.length)
            for item in candidate:
                support *= self.data[item]
            agg_support = support.sum() / self.length
            if agg_support >= self.min_support:
                frequent_k_itemsets.append(candidate)
            agg_support_list.append(agg_support)
        max_support = max(agg_support_list)
        for i, support in enumerate(agg_support_list):
            if support >= self.min_support / max_support:
                frequent_k_itemsets_candidates.append(candidates[i])

        return frequent_k_itemsets, frequent_k_itemsets_candidates

    # @staticmethod
    # def generate_candidates(prev_itemsets, k) -> list:
    #     candidates = list()
    #
    #     for itemset1 in tqdm(prev_itemsets):
    #         prev_itemsets.remove(itemset1)
    #         for itemset2 in prev_itemsets:
    #             if k - 1 == 1:
    #                 union_set = {itemset1, }.union({itemset2, })
    #             else:
    #                 union_set = set(itemset1).union(set(itemset2))
    #             if len(union_set) == k and union_set not in candidates:
    #                 candidates.append(union_set)
    #     return candidates

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
