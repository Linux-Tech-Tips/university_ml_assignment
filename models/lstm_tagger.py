# First Candidate Model - Bidirectional LSTM tagger

import numpy as np
import pandas as pd
import torch
from torch import nn

from model import Model
from vocab import Vocab
from metrics import Metrics


# TODO Architecture
# -> PyTorch nn.Module-extending BiLSTM class implementing a Bidirectional LSTM model with customisable params, with a forward, train and test loops
# -> BiLSTMTagger class which uses a BiLSTM class model, which includes a forward_pass impl calling the BiLSTM, a train function for N epochs with K-fold cross validation,
#    as well as a function running train-val on a hyperparameter matrix to find optimal hyperparameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMTagger(Model):

    def __init__(self, dataset: pd.DataFrame, learning_rate: float = 10**(-3), embedding_dim: int = 100, hidden_dim: int = 256, dropout_p: int = 0.4, lstm_layers: int = 2):
        self.vocab = Vocab(dataset[dataset["subset"] != "test"]) # Only building a vocabulary of non-test data to prevent leakage
        self.model = BiLSTM(self.vocab.length(), self.vocab.tag_count(), embedding_dim, hidden_dim, dropout_p, lstm_layers).to(DEVICE)
        self.learning_rate = learning_rate
        print(f"Model Log: Initialised BiLSTMTagger Model on device {DEVICE}")

    def forward_pass(self, inputs: list) -> list:
        input_ids = torch.tensor([self.vocab.get_token_id(x) for x in inputs], device=DEVICE)
        output = self.model(input_ids)
        output_ids = torch.argmax(output, 1)
        return [self.vocab.get_tag_by_id(x) for x in output_ids]

    def fit(self, dataset: pd.DataFrame, epochs: int):

        print("Model Log: Starting Fitting")

        # Process sentences and tags into training-ready format format
        sentences = {fold_id: [torch.tensor([self.vocab.get_token_id(x) for x in sentence], device=DEVICE) for sentence in dataset[dataset["subset"].eq("train") & dataset["fold_id"].eq(fold_id)]["tokenized_text"][:5000]] for fold_id in set(dataset["fold_id"]) if fold_id >= 0}
        tags = {fold_id: [torch.tensor([self.vocab.get_tag_id(x) for x in tag], device=DEVICE) for tag in dataset[dataset["subset"].eq("train") & dataset["fold_id"].eq(fold_id)]["tokenized_tags"][:5000]] for fold_id in set(dataset["fold_id"]) if fold_id >= 0}
        print("Model Log: Processed Sentences/Tags to IDs")

        # Initialise loss function, optimiser and metrics reporting
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        metrics = Metrics(self, dataset)

        for epoch in range(epochs):
            print(f"Model Log: Running epoch {epoch}...")
            accuracy = []
            f1 = []
            for fold in sentences:
                self.model.train_epoch(sentences[fold], tags[fold], loss_fn, optimizer)
                print(f"Model Log: Running with CV on fold {fold} done!")
                accuracy.append(metrics.get_val_accuracy(fold, True)[0])
                f1.append(metrics.get_val_macro_f1(fold, True)[0])
            print(f"Model Log: Epoch {epoch} done!")
            print("Model Log: Score:")
            print(f" -> Accuracy: mean {np.mean(accuracy)}, standard deviation {np.std(accuracy)}")
            print(f" -> Macro-F1: mean {np.mean(f1)}, standard deviation {np.std(f1)}")


class BiLSTM(nn.Module):

    def __init__(self, num_terms: int, num_tags: int, embedding_dim: int, hidden_dim: int, dropout_p: int = 0.5, lstm_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(num_terms, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, bidirectional=True, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(2 * hidden_dim, num_tags)

    # TODO Add hyperparameter-finding function that runs through matrix and selects optimal ones, run it in notebook

    def forward(self, x: torch.tensor):
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

    def train_epoch(self, sentences: list[torch.tensor], tags: list[torch.tensor], loss_fn, optimizer):
        """Expects training data as a list of lists containing token IDs and corresponding list of lists with a tag ID for each token ID"""

        # 1. set model to train mode
        self.train()
        # 2. loop over training data
        assert(len(sentences) == len(tags))
        for sentence, tag in zip(sentences, tags):
            # get prediction with current model
            prediction = self(sentence)
            # get loss with the given loss function
            loss = loss_fn(prediction, tag)

            # backward pass to accumulate loss function gradients
            loss.backward()
            # perform single optimiser step with loss gradients
            optimizer.step()
            # reset gradients for next training step
            optimizer.zero_grad()

        # 3. set model to eval mode after epoch done
        self.eval()

