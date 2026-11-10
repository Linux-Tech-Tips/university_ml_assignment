# First Candidate Model - Bidirectional LSTM tagger

import numpy as np
import pandas as pd
import torch
from torch import nn

from model import Model
from vocab import Vocab
from metrics import Metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMTagger(Model):

    def __init__(self, dataset: pd.DataFrame, excluded_fold: int = -1, learning_rate: float = 10**(-3), embedding_dim: int = 100, hidden_dim: int = 256, dropout_p: int = 0.4, lstm_layers: int = 2):
        vocab_set = dataset[dataset["subset"].ne("test") & dataset["fold_id"].ne(excluded_fold)] # Excluding test and optionally fold data from vocabulary to prevent leakage
        self.vocab = Vocab(vocab_set)
        self.model = BiLSTM(self.vocab.length(), self.vocab.tag_count(), embedding_dim, hidden_dim, dropout_p, lstm_layers).to(DEVICE)
        self.learning_rate = learning_rate
        print(f"Model Log: Initialised BiLSTMTagger Model on device {DEVICE}")

    def forward_pass(self, inputs: list) -> list:
        input_ids = torch.tensor([self.vocab.get_token_id(x) for x in inputs], device=DEVICE)
        # Run model inference in eval mode and without computing gradients (no_autograd)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids)
        output_ids = torch.argmax(output, 1)
        return [self.vocab.get_tag_by_id(x) for x in output_ids]

    def fit(self, dataset: pd.DataFrame, epochs: int, fraction: float = 1.0, fold: int = -1):
        print("Model Log: Starting Fitting")

        # Optionally reduce fraction of training data by taking random sample of non-test data
        # for computational reasons and for quicker debugging
        if fraction < 1.0:
            dataset_frac = dataset[dataset["subset"] != "test"].sample(frac=fraction)
        else:
            dataset_frac = dataset

        # Process sentences and tags into training-ready format format
        if fold >= 0:
            sentences = [torch.tensor([self.vocab.get_token_id(x) for x in sentence], device=DEVICE)
                        for sentence in dataset_frac[dataset_frac["subset"].eq("train") & dataset_frac["fold_id"].ne(fold)]["tokenized_text"]]
            tags = [torch.tensor([self.vocab.get_tag_id(x) for x in tag], device=DEVICE)
                    for tag in dataset_frac[dataset_frac["subset"].eq("train") & dataset_frac["fold_id"].ne(fold)]["tokenized_tags"]]
        else:
            sentences = [torch.tensor([self.vocab.get_token_id(x) for x in sentence], device=DEVICE)
                        for sentence in dataset_frac[dataset_frac["subset"].eq("train")]["tokenized_text"]]
            tags = [torch.tensor([self.vocab.get_tag_id(x) for x in tag], device=DEVICE)
                    for tag in dataset_frac[dataset_frac["subset"].eq("train")]["tokenized_tags"]]
        print("Model Log: Processed Sentences/Tags to IDs")

        # Initialise loss function, optimiser and metrics reporting
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        metrics = Metrics(self, dataset)

        # Train over N epochs excluding a random cross-validation fold
        acc = None
        f1 = None
        for epoch in range(epochs):
            print(f"Model Log: Running epoch {epoch}...")
            self.model.train_epoch(sentences, tags, loss_fn, optimizer)
            acc, _, _ = metrics.get_val_accuracy(fold, True)
            f1, _ = metrics.get_val_macro_f1(fold, True)
            print(f"Model Log: Epoch {epoch} done!")
            print("Model Score:")
            print(f" -> Accuracy: {acc}")
            print(f" -> Macro-F1: {f1}")
        return acc, f1


    @staticmethod
    def _tune_param(dataset: pd.DataFrame, model_params: list[dict], epochs: int, fraction: float, folds: list):
        optimal = None
        best_score = 0
        best_score_std = None
        table = []
        # Go through each set of model params provided
        for param_set in model_params:
            f1s = []
            # Gather F1 accuracy across all 5 folds
            for fold in folds:
                model = BiLSTMTagger(dataset, **param_set)
                f1s.append(model.fit(dataset, epochs, fraction, fold)[1])
            # Get mean and standard deviation of accuracies with each of the folds excluded
            mean_f1 = np.mean(f1s)
            stdev_f1 = np.std(f1s)
            table.append((param_set, mean_f1, stdev_f1))
            # Update findings if better score obtained
            if mean_f1 > best_score:
                optimal = param_set
                best_score = mean_f1
                best_score_std = stdev_f1
        return optimal, best_score, best_score_std, table

    @staticmethod
    def tune(dataset: pd.DataFrame, epochs: int, fraction: float, folds: list,
            learning_rates: list[float], embedding_dims: list[int], hidden_dims: list[float], dropout_ps: list[float], lstm_layers: list[int]):

        # Tune in stages for compute efficiency rather than grid -> learning rate, then dropout, then layers and dims
        # Tune on F1 portion of the score to account for long-tail distributions

        print(f"Model Log: Hyperparameter tuning started (fraction {fraction}, on {epochs} epochs)")
        all_table = {}

        # 1. Tune learning rate
        optimal_lr, best_lr_f1, best_lr_std, table = BiLSTMTagger._tune_param(dataset, [{"learning_rate": x} for x in learning_rates], epochs, fraction, folds)
        print(f"Model Log: Found best learning rate of {optimal_lr} with score {best_lr_f1} and standard deviation {best_lr_std}")
        all_table["learning_rate"] = table

        # 2. Tune dropout percentage
        optimal_dropout, best_dropout_f1, best_dropout_std, table = BiLSTMTagger._tune_param(dataset, [{"dropout_p": x} for x in dropout_ps], epochs, fraction, folds)
        print(f"Model Log: Found best dropout probability of {optimal_dropout} with score {best_dropout_f1} and standard deviation {best_dropout_std}")
        all_table["dropout_p"] = table

        # 3. Tune embedding dimensions
        optimal_embedding, best_emb_f1, best_emb_std, table = BiLSTMTagger._tune_param(dataset, [{"embedding_dim": x} for x in embedding_dims], epochs, fraction, folds)
        print(f"Model Log: Found best embedding dimension of {optimal_embedding} with score {best_emb_f1} and standard deviation {best_emb_std}")
        all_table["embedding_dim"] = table

        # 4. Tune hidden dimensions and LSTM layers grid
        lstm_grid = [{"hidden_dim": hidden, "lstm_layers": layers} for hidden in hidden_dims for layers in lstm_layers]
        optimal_lstm, best_lstm_f1, best_lstm_std, table = BiLSTMTagger._tune_param(dataset, lstm_grid, epochs, fraction, folds)
        print(f"Model Log: Found best LSTM params of {optimal_lstm} with score {best_lstm_f1} and standard deviation {best_lstm_std}")
        all_table["lstm"] = table

        return {"learning_rate": (optimal_lr, best_lr_f1, best_lr_std),
                "dropout_p": (optimal_dropout, best_dropout_f1, best_dropout_std),
                "embedding_dim": (optimal_embedding, best_emb_f1, best_emb_std),
                "lstm_params": (optimal_lstm, best_lstm_f1, best_lstm_std),
                "case_table": all_table}


class BiLSTM(nn.Module):

    def __init__(self, num_terms: int, num_tags: int, embedding_dim: int, hidden_dim: int, dropout_p: int = 0.5, lstm_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(num_terms, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, bidirectional=True, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(2 * hidden_dim, num_tags)

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

