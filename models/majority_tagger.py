# Baseline Model - Majority-class Tagger

import pandas as pd

from model import Model
from metrics import Metrics


class MajorityTagger(Model):

    def __init__(self):
        self.tags = {}
        self.most_common = "N/A"

    def forward_pass(self, inputs: list) -> list:
        outputs = []
        for token in inputs:
            if token in self.tags:
                outputs.append(self.tags[token])
            else:
                outputs.append(self.most_common)
        return outputs

    def fit(self, dataset: pd.DataFrame, fold: int = -1):
        """Runs the majority tagger training excluding a given fold, or on the whole training set if 'fold' is '-1' (default)"""

        # reset before new call to fit
        self.tags = {}
        self.most_common = "N/A"

        # select training data subset and run training
        if fold >= 0:
            train_data = dataset[dataset["subset"].eq("train") & dataset["fold_id"].ne(fold)]
        else:
            train_data = dataset[dataset["subset"].eq("train")]
        self._train(train_data)

    def _train(self, dataset: pd.DataFrame):
        state = {}
        # run training step for each row
        for row_idx in dataset.index:
            row_data = dataset.loc[row_idx]
            self._train_step(row_data["tokenized_text"], row_data["tokenized_tags"], state)
        # finalise most frequent tag from state
        for word in state:
            majority_tag = max(state[word], key=state[word].get)
            self.tags[word] = majority_tag
        # assign most common tag vased on processed tags dict
        self.most_common = max(set(self.tags.values()), key=list(self.tags.values()).count)

    def _train_step(self, text: list, tags: list, state: dict):
        if len(text) != len(tags):
            raise Exception("ERROR: Majority-class Tagger Model training got diverging text and tag lengths")
        for pair in zip(text, tags):
            word = pair[0]
            tag = pair[1]
            if word in state:
                if tag in state[word]:
                    state[word][tag] += 1
                else:
                    state[word][tag] = 1
            else:
                state[word] = {}
                state[word][tag] = 1

