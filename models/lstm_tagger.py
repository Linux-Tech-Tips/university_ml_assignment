# First Candidate Model - Bidirectional LSTM tagger

import pandas as pd
import torch
from torch import nn

from model import Model
from vocab import Vocab


# TODO Architecture
# -> PyTorch nn.Module-extending BiLSTM class implementing a Bidirectional LSTM model with customisable params, with a forward, train and test loops
# -> BiLSTMTagger class which uses a BiLSTM class model, which includes a forward_pass impl calling the BiLSTM, a train function for N epochs with K-fold cross validation,
#    as well as a function running train-val on a hyperparameter matrix to find optimal hyperparameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEBUG: Loaded LSTM tagger for device", DEVICE)

class BiLSTMTagger(Model):

    def __init__(self, dataset: pd.DataFrame, learning_rate: float = 10**(-3), embedding_dim: int = 100, hidden_dim: int = 256, dropout_p: int = 0.4, lstm_layers: int = 2):
        self.vocab = Vocab(dataset)
        self.model = BiLSTM(self.vocab.length(), self.vocab.tag_count(), embedding_dim, hidden_dim, dropout_p, lstm_layers).to(DEVICE)
        self.learning_rate = learning_rate

    def forward_pass(self, inputs: list) -> list:
        input_ids = [self.vocab.get_token_id(x) for x in inputs]
        output = self.model(input_ids)
        output_ids = torch.argmax(output, 1)
        return [self.vocab.get_tag_by_id(x) for x in output_ids]

    def fit(self, dataset: pd.DataFrame, epochs: int):

        print("DEBUG: Starting fit")
        sentences = [[self.vocab.get_token_id(x) for x in sentence] for sentence in dataset[dataset["subset"] == "train"]["tokenized_text"]]
        print("DEBUG: Processed sentences to ID")
        tags = [[self.vocab.get_tag_id(x) for x in tag] for tag in dataset[dataset["subset"] == "train"]["tokenized_tags"]]
        print("DEBUG: Processed tags to ID")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(epochs):
            print("DEBUG: Running epoch...")
            self.model.train_epoch(sentences, tags, loss_fn, optimizer)
            print(f"Epoch {epoch} done!")


class BiLSTM(nn.Module):

    def __init__(self, num_terms: int, num_tags: int, embedding_dim: int, hidden_dim: int, dropout_p: int = 0.5, lstm_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(num_terms, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, bidirectional=True, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(2 * hidden_dim, num_tags)

    def forward(self, x: list):
        # turn input list into PyTorch-friendly tensor
        x = torch.tensor(x, device=DEVICE)
        # 1. embed the input IDs
        embedded = self.embedding(x)
        # 2. dropout and prepare input for LSTM layer
        lstm_in = self.dropout(embedded).unsqueeze(0)
        # 3. run LSTM layer and dropout
        h, _ = self.lstm(lstm_in)
        lstm_out = self.dropout(h)
        # 4. run linear to get output tag probabilities
        network_out = self.out(lstm_out).squeeze(0)
        return network_out

    def train_epoch(self, sentences: list, tags: list, loss_fn, optimizer):
        """Expects training data as a list of lists containing token IDs and corresponding list of lists with a tag ID for each token ID"""

        # 1. set model to train mode
        self.train()
        # 2. loop over training data
        assert(len(sentences) == len(tags))
        for sentence, tag in zip(sentences, tags):
            # get prediction with current model
            prediction = self(sentence)
            # get loss with the given loss function
            loss = loss_fn(prediction, torch.tensor(tag, device=DEVICE))

            # backward pass to accumulate loss function gradients
            loss.backward()
            # perform single optimiser step with loss gradients
            optimizer.step()
            # reset gradients for next training step
            optimizer.zero_grad()

        # 3. set model to eval mode after epoch done
        self.eval()

