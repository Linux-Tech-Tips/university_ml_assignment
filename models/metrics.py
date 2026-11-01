# Measure performance metrics on an arbitrary model

import pandas as pd

from model import Model

# TODO Architecture:
# -> given dataset DataFrame and Model instance, be able to get metrics for various things
# -> save a reference to dataset and Model, have methods to get metrics
# -> method to get metrics on test set, method to get metrics on Nth fold
# -> set of methods for each metric
# -> method for accuracy: arithmetic mean of 1 for right prediction, 0 for wrong
# -> macro F1:
#   -> (2TP) / (2TP + FP + FN)
#   -> TP .. true positive, count given tag in both prediction and golden
#   -> FP .. false positive, count given tag in prediction but not golden
#   -> FN .. false negative, count given tag in golden but not prediction
# -> do counts for all sentences in given set, then scores

class Metrics:

    def __init__(self, model: Model, dataset: pd.DataFrame):
        """Expects a concrete Model instance and a preprocessed dataset"""
        self.model = model
        self.dataset = dataset

    def get_val_accuracy(self, fold: int):
        subset = self._get_val_subset(fold)
        accuracy, correct_count, total_count = self._get_accuracy(subset)
        print(f"Metrics: accuracy {accuracy:.4f} ({correct_count}/{total_count} tagged correct) on {fold}-th fold")

    def get_val_macro_f1(self, fold: int):
        subset = self._get_val_subset(fold)

    def get_test_accuracy(self):
        subset = self._get_test_subset()
        accuracy, correct_count, total_count = self._get_accuracy(subset)
        print(f"Metrics: accuracy {accuracy:.4f} ({correct_count}/{total_count} tagged correct) on test set")

    def get_test_macro_f1(self):
        subset = self._get_test_subset()


    def _get_val_subset(self, fold: int) -> pd.DataFrame:
        return self.dataset[self.dataset["subset"].eq("train") & self.dataset["fold_id"].eq(fold)]

    def _get_test_subset(self) -> pd.DataFrame:
        return self.dataset[self.dataset["subset"].eq("test")]

    def _get_accuracy(self, dataset: pd.DataFrame):
        correct_count = 0
        total_count = 0
        for i in dataset.index:
            in_text = dataset.loc[i]["tokenized_text"]
            golden_tags = dataset.loc[i]["tokenized_tags"]
            result = self.model.forward_pass(in_text)
            # Error if model returns unexpected length
            if len(result) != len(golden_tags):
                raise Exception("ERROR: Model returned unexpected number of tags")
            for tags in zip(golden_tags, result):
                total_count += 1
                if tags[0] == tags[1]:
                    correct_count += 1
        accuracy = correct_count / total_count
        return (accuracy, correct_count, total_count)

    def _get_macro_f1(self, dataset: pd.DataFrame):
        pass
