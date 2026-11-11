# First Candidate Model - Bidirectional LSTM tagger

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from model import Model
from vocab import Vocab
from metrics import Metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG_PAD = -100

# Utility function to collate token id/tag id data in a given batch, for efficient batched training
def collate_id_tag_data(data_batch):
    token_ids, tag_ids = zip(*data_batch)
    lengths = torch.tensor([len(x) for x in token_ids])
    return (pad_sequence(token_ids, batch_first=True, padding_value=0),
            pad_sequence(tag_ids, batch_first=True, padding_value=TAG_PAD),
            lengths)


class BiLSTMTagger(Model):

    def __init__(self, dataset: pd.DataFrame, excluded_fold: int = -1, learning_rate: float = 10**(-3), embedding_dim: int = 100, hidden_dim: int = 256, dropout_p: int = 0.4, lstm_layers: int = 2):
        # Excluding test and optionally fold data from vocabulary to prevent leakage
        if excluded_fold >= 0:
            vocab_set = dataset[dataset["subset"].ne("test") & dataset["fold_id"].ne(excluded_fold)]
        else:
            vocab_set = dataset[dataset["subset"].ne("test")]
        self.vocab = Vocab(vocab_set)
        self.model = BiLSTM(self.vocab.length(), self.vocab.tag_count(), embedding_dim, hidden_dim, dropout_p, lstm_layers).to(DEVICE)
        self.learning_rate = learning_rate
        print(f"Model Log: Initialised BiLSTMTagger Model on device {DEVICE}")

    def forward_pass(self, inputs: list) -> list:
        input_ids = torch.tensor([self.vocab.get_token_id(x) for x in inputs], device=DEVICE).unsqueeze(0)
        # Run model inference in eval mode and without computing gradients (no_autograd)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids, torch.tensor([len(inputs)]))
        output_ids = torch.argmax(output, -1).squeeze(0)
        return [self.vocab.get_tag_by_id(x) for x in output_ids]

    def fit(self, dataset: pd.DataFrame, epochs: int, fraction: float = 1.0, fold: int = -1, batch_size: int = 64):
        print("Model Log: Starting Fitting")

        # Optionally reduce fraction of training data by taking random sample of non-test data
        # for computational reasons and for quicker debugging
        if fraction < 1.0:
            dataset_frac = dataset[dataset["subset"] != "test"].sample(frac=fraction, random_state=261110)
        else:
            dataset_frac = dataset

        # Process sentences and tags into training-ready format format
        if fold >= 0:
            sentences = [torch.tensor([self.vocab.get_token_id(x) for x in sentence])
                        for sentence in dataset_frac[dataset_frac["subset"].eq("train") & dataset_frac["fold_id"].ne(fold)]["tokenized_text"]]
            tags = [torch.tensor([self.vocab.get_tag_id(x) for x in tag], device=DEVICE)
                    for tag in dataset_frac[dataset_frac["subset"].eq("train") & dataset_frac["fold_id"].ne(fold)]["tokenized_tags"]]
        else:
            sentences = [torch.tensor([self.vocab.get_token_id(x) for x in sentence])
                        for sentence in dataset_frac[dataset_frac["subset"].eq("train")]["tokenized_text"]]
            tags = [torch.tensor([self.vocab.get_tag_id(x) for x in tag], device=DEVICE)
                    for tag in dataset_frac[dataset_frac["subset"].eq("train")]["tokenized_tags"]]
        print("Model Log: Processed Sentences/Tags to IDs")
        data = [(sentences[i], tags[i]) for i in range(len(sentences))]
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_id_tag_data)

        # Initialise loss function, optimiser and metrics reporting
        loss_fn = nn.CrossEntropyLoss(ignore_index=TAG_PAD)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        metrics = Metrics(self, dataset_frac)

        # Train over N epochs excluding a random cross-validation fold
        acc = None
        f1 = None
        for epoch in range(epochs):
            print(f"Model Log: Running epoch {epoch}...")
            self.model.train_epoch(loader, loss_fn, optimizer)
            acc, _, _ = metrics.get_val_accuracy(fold if fold >= 0 else 0, True)
            f1, _ = metrics.get_val_macro_f1(fold if fold >= 0 else 0, True)
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

    def forward(self, x: torch.tensor, lengths: torch.tensor):
        # 1. embed the input IDs
        embedded = self.embedding(x)
        # 2. dropout and prepare input for LSTM layer
        lstm_in = pack_padded_sequence(self.dropout(embedded), lengths, batch_first=True, enforce_sorted=False)
        # 3. run LSTM layer, unpack and dropout
        packed_lstm_out, _ = self.lstm(lstm_in)
        padded_lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True, total_length=x.size(1))
        lstm_out = self.dropout(padded_lstm_out)
        # 4. run linear to get output tag probabilities
        network_out = self.out(lstm_out)
        return network_out

    def train_epoch(self, loader: DataLoader, loss_fn, optimizer):
        """Expects training data as a list of lists containing token IDs and corresponding list of lists with a tag ID for each token ID"""

        # 1. set model to train mode
        self.train()
        # 2. loop over training data
        for sentences, tags, lengths in loader:
            # move sentences and tags tensors to GPU if active for efficient processing
            sentences = sentences.to(DEVICE)
            tags = tags.to(DEVICE)
            # get prediction with current model
            prediction = self(sentences, lengths)
            # get loss with the given loss function
            loss = loss_fn(prediction.reshape(-1, prediction.size(-1)), tags.reshape(-1))

            # backward pass to accumulate loss function gradients
            loss.backward()
            # perform single optimiser step with loss gradients
            optimizer.step()
            # reset gradients for next training step
            optimizer.zero_grad()

        # 3. set model to eval mode after epoch done
        self.eval()

