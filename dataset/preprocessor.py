# Data Pre-processing and Splitting Pipeline

import math
from random import Random

import pandas as pd

class DatasetPipeline:
    """Dataset Pre-processing pipeline with steps and constants derived from the attached dataset audit"""

    _RANDOM_SEED = 20261031
    RANDOM = Random(_RANDOM_SEED)

    PROCESSED_COLUMNS = ["raw_text", "label", "subset", "fold_id", "tokenized_text", "tokenized_tags"]

    NUM_FOLDS = 5

    TRAIN_LABEL = "train"
    TEST_LABEL = "test"


    def __init__(self, dataset: pd.DataFrame):
        self.original = dataset
        self.processed = pd.DataFrame(columns=DatasetPipeline.PROCESSED_COLUMNS)
        self._split_dataset()
        self._preprocess_dataset()

    def get(self) -> pd.DataFrame:
        return self.processed


    def _split_dataset(self):
        for label in set(self.original["label"]):
            # stratification - do an 80-20 split (+CV) for each label individually (cast to 'set' first removes duplicates)
            text_list = list(set(self.original[self.original["label"] == label]["raw_text"]))

            # initial 80:20 train/test split
            test = DatasetPipeline.RANDOM.sample(text_list, int(0.2 * len(text_list)))
            train = [x for x in text_list if x not in test]

            # K-fold CV split into validation folds
            DatasetPipeline.RANDOM.shuffle(train)
            fold_length = math.ceil(len(train) / DatasetPipeline.NUM_FOLDS)
            if len(train) % fold_length == 0 and len(train) // fold_length != DatasetPipeline.NUM_FOLDS:
                folds = [train[i:(i+fold_length)] for i in range(0, (len(train) - fold_length), fold_length)]
                remainder_idx = fold_length * (DatasetPipeline.NUM_FOLDS-2)
                folds.append(train[remainder_idx:(remainder_idx + fold_length // 2)])
                folds.append(train[(remainder_idx + fold_length // 2):])
            else:
                folds = [train[i:(i+fold_length)] for i in range(0, len(train), fold_length)]

            rows = []
            # add test segment rows
            for text in test:
                rows.append([text, label, DatasetPipeline.TEST_LABEL, -1, None, None])
            # add train/cv fold rows
            for i in range(len(folds)):
                for text in folds[i]:
                    rows.append([text, label, DatasetPipeline.TRAIN_LABEL, i, None, None])
            # add rows to 'processed' dataframe
            self.processed = pd.concat([self.processed, pd.DataFrame(rows, columns=DatasetPipeline.PROCESSED_COLUMNS)], ignore_index=True)

    def _preprocess_dataset(self):
        for i in range(len(self.processed)):
            # custom tokenisation of the raw text
            raw_text = self.processed.loc[i]["raw_text"]
            tokens = [(word.rsplit("/", 1)[0], word.rsplit("/", 1)[1]) for word in raw_text.split(" ")]
            tokenized_text = []
            tokenized_tags = []
            for token in tokens:
                tokenized_text.append(token[0].strip())
                # modifying the POS tags as specified in the dataset audit
                tag = token[1].strip().replace("fw-", "").replace("-tl", "").replace("-hl", "").replace("-nc", "").replace("-n", "")
                if "+" in tag:
                    tag = tag.split("+")[0]
                tokenized_tags.append(tag)
            self.processed.at[i, "tokenized_text"] = tokenized_text
            self.processed.at[i, "tokenized_tags"] = tokenized_tags

