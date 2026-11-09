# ID-based token vocabulary as well as tag collection for ID-based models

import pandas as pd

class Vocab:

    def __init__(self, dataset: pd.DataFrame):

        # Processing vocabulary
        vocab = set()
        for sentence in dataset["tokenized_text"]:
            for word in sentence:
                vocab.add(word)
        self.vocab = ["UNK"] + sorted(list(vocab)) # 'UNK' added as token 0 to mark unknown words
        self._vocab_map = {self.vocab[i]: i for i in range(len(self.vocab))} # dict-based maps for efficient token id lookups

        # Processing POS tag types
        tagset = set()
        for tags in dataset["tokenized_tags"]:
            for tag in tags:
                tagset.add(tag)
        self.tags = sorted(list(tagset))
        self._tags_map = {self.tags[i]: i for i in range(len(self.tags))}

    def length(self):
        return len(self.vocab)

    def tag_count(self):
        return len(self.tags)


    def get_token_id(self, token: str):
        if token in self._vocab_map:
            return self._vocab_map[token]
        return 0

    def get_token_by_id(self, token_id: int):
        if token_id < self.length():
            return self.vocab[token_id]
        return self.vocab[0]

    def get_tag_id(self, tag: str):
        if tag in self._tags_map:
            return self._tags_map[tag]
        return None

    def get_tag_by_id(self, tag_id: int):
        if tag_id < self.tag_count():
            return self.tags[tag_id]
        return None
