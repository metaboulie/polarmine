# Datasets
BLACK_FRIDAY_DATASET_PATH = "./src/BlackFridaytrain.csv"

# Association Analysis
ASSOCIATION_MIN_SUPPORT = 0.1
ASSOCIATION_MIN_CONFIDENCE = 0.4
ASSOCIATION_RULE = "confidence"
ASSOCIATION_MIN_VALUE = {
  "confidence": 0.4,
  "support": 0.01,
  "lift": 3,
  "leverage": 3,
  "antecedent_support": 0.05,
  "consequent_support": 0.05,
}
ITEMSET_LENGTH = 2
