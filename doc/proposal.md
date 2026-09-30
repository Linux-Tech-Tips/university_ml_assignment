---
geometry: margin=0.8in
fontsize: 10pt
...

# CT4101 Machine Learning Project Proposal

**Project Title:** Analysis of sequence-processing Machine Learning techniques applied to the context-aware part-of-speech tagging natural language classification problem.

**Author:** Martin Klacer, University of Galway

**Project Repository:** [https://github.com/Linux-Tech-Tips/university_ml_assignment](https://github.com/Linux-Tech-Tips/university_ml_assignment)

## Task

**Task Type:** Classification

**Task Description:** Part-of-speech Tagging - multi-class identification of part-of-speech for each word in a given sequence of text

**Target Variable:** Part-of-speech of a given word in a given sequence of text

**Main Input Features:** Current token as well as preceeding and following tokens in the sequence, where a token maps approximately to one word using some tokenization algorithm.

## Dataset

**Selected Dataset:** the Brown University Standard Corpus of Present-Day American English (*or simply* Brown Corpus)

**Dataset Provided By:** NLTK (Natural Language Tool Kit) Platform, by the NLTK Project, University of Pennsylvania

**Dataset Accessible Through:** Kaggle Datasets at [https://www.kaggle.com/datasets/nltkdata/brown-corpus](https://www.kaggle.com/datasets/nltkdata/brown-corpus)

**Dataset License/Usability:** As per the NLTK dataset licensing information ([https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md)),
the Brown Corpus can be used for educational purposes, with usage being restricted to non-commercial purposes.
Confirmed through the linguistics group ICAME (International Computer Archive of Modern and Medieval English), who are the distributor of the corpus,
that academic use is permitted ([https://icame.info/icame-corpora/](https://icame.info/icame-corpora/)).

**Dataset Citation:**

> Francis, W. N., and H. Kučera. Manual of Information to Accompany a Standard Sample of Present-day Edited American English, for Use with Digital Computers.
> Original ed. 1964, revised 1971, revised and augmented 1979. Providence, R.I.: Department of Linguistics, Brown University.

**Approx. Number of Records:** 500 sequences (samples of English language content) each with >2000 words, with a total of 1,014,312 words (all tagged).

**Important Data Risks:** The sequences provided come from multiple genres, which may influence word usage and the corresponding POS tag.
To ensure appropriate data quality for training, an appropriate genre distribution needs to be considered.

**Dataset Split:** The Brown Corpus is a single collection, meaning a custom train/validation/test dataset split will be required.

## Models & Evaluation

**Proposed Baseline Model:** Majority-class literal word POS tagger - assign the tag that occurs most frequently for any given word in the dataset

**Candidate Models:**

1. Bidirectional Long Short-Term Memory (Bi-LSTM), stateful recurrent neural network used for sequence processing with potential cross-element dependencies
2. Encoder-only Transformer (similar to BERT), an architecture using non-masked self-attention followed by a FNN-like layer to process sequences, using attention to model context

**Key Evaluation Metric:** Accuracy on the test split of the dataset - accuracy defined as percentage of words tagged correctly by the model.

**Supplementary Evaluation Metrics:** Zero-shot human-generated example accuracy can be used for demonstration/supplementary purposes.

## Privacy & Fairness Statement

While transformer-based language models are used in a variety of ways in recent years, one of the objectives of this portfolio is to highlight
some non-generative uses of machine learning and the transformer architecture, specifically in the area of natural language processing and computational linguistics,
by demonstrating the effectiveness of the self-attention mechanism for an example natural language sequence processing task (part-of-speech tagging).
To ensure privacy and fairness, the dataset used for this is an at this point decades old corpus of various language examples, prepared by Brown University for academic purposes,
rather than any sort of data scraped from online sources, which could constitute misuse or training on unauthorised/potentially sensitive information.

## AI Usage Statement

For the purposes of this part of the portfolio, generative AI (Anthropic Claude) was used as a supplementary search tool to explore dataset options.
All information in this assignment was independently verified from credible online sources, and all content in this document was human-authored.

