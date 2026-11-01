# Measure performance metrics on an arbitrary model

import pandas as pd

from model import Model


class Metrics:

    def __init__(self, model: Model, dataset: pd.DataFrame):
        """Expects a concrete Model instance and a preprocessed dataset"""
        self.model = model
        self.dataset = dataset

    def get_val_accuracy(self, fold: int, quiet: bool = False):
        subset = self._get_val_subset(fold)
        accuracy, correct_count, total_count = self._get_accuracy(subset)
        if not quiet:
            print(f"Metrics: accuracy {accuracy:.4f} ({correct_count}/{total_count} tagged correct) on {fold}-th fold")
        return accuracy, correct_count, total_count

    def get_val_macro_f1(self, fold: int, quiet: bool = False):
        subset = self._get_val_subset(fold)
        macro_f1, scores = self._get_macro_f1(subset)
        if not quiet:
            print(f"Metrics: Macro-F1 {macro_f1:.4f}")
        return macro_f1, scores

    def get_test_accuracy(self, quiet: bool = False):
        subset = self._get_test_subset()
        accuracy, correct_count, total_count = self._get_accuracy(subset)
        if not quiet:
            print(f"Metrics: accuracy {accuracy:.4f} ({correct_count}/{total_count} tagged correct) on test set")
        return accuracy, correct_count, total_count

    def get_test_macro_f1(self, quiet: bool = False):
        subset = self._get_test_subset()
        macro_f1, scores = self._get_macro_f1(subset)
        if not quiet:
            print(f"Metrics: Macro-F1 {macro_f1:.4f}")
        return macro_f1, scores


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
        tag_tp = {}
        tag_fp = {}
        tag_fn = {}
        # Get counts
        for i in dataset.index:
            in_text = dataset.loc[i]["tokenized_text"]
            golden_tags = dataset.loc[i]["tokenized_tags"]
            result = self.model.forward_pass(in_text)
            # Error if model returns unexpected length
            if len(result) != len(golden_tags):
                raise Exception("ERROR: Model returned unexpected number of tags")
            for tags in zip(golden_tags, result):
                if tags[0] == tags[1]:
                    # true positive
                    if tags[0] in tag_tp:
                        tag_tp[tags[0]] += 1
                    else:
                        tag_tp[tags[0]] = 1
                else:
                    golden = tags[0]
                    predicted = tags[1]
                    # golden tag false negative
                    if golden in tag_fn:
                        tag_fn[golden] += 1
                    else:
                        tag_fn[golden] = 1
                    # predicted tag false positive
                    if predicted in tag_fp:
                        tag_fp[predicted] += 1
                    else:
                        tag_fp[predicted] = 1
        # Calculate F1 score for each tag
        scores = {}
        all_tags = set(tag_tp.keys()).union(tag_fp.keys()).union(tag_fn.keys())
        for tag in all_tags:
            tp = tag_tp.get(tag, 0)
            fp = tag_fp.get(tag, 0)
            fn = tag_fn.get(tag, 0)
            f1_score = (2*tp) / (2*tp + fp + fn)
            scores[tag] = f1_score
        # Calculate macro-F1 (arithmetic mean of F1 scores of each class)
        macro_f1 = sum(scores.values()) / len(scores.values())
        return (macro_f1, scores)

