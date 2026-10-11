from kagglehub import KaggleDatasetAdapter
import kagglehub

def load_dataset_df(kaggle_path: str, csv_file: str):
    """Loads any CSV file from a dataset from Kaggle (using kaggle_path) into a Pandas Dataframe"""
    return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, kaggle_path, csv_file)

def load_brown_corpus_df():
    """Loads the Brown Corpus CSV as a Pandas Dataframe"""
    return load_dataset_df("nltkdata/brown-corpus", "brown.csv")

