---
geometry: margin=1in
...

# CT4101 Machine Learning Project Proposal

**Martin Klacer, University of Galway**

**Academic Year 2026/27**

**CT4101 Machine Learning**

This document outlines the proposal for my Machine Learning Portfolio Project,
mainly concerning the dataset, task and intended evaluation.

## Task

**Task Type:** Classification

**Task Description:** Part-of-speech Tagging - multi-class identification of part-of-speech for each word in a given sequence of text

**Variables:** Both current word content and surrounding text in the rest of the sequence determine the class of the current word

## Dataset

**Selected Dataset:** the Brown University Standard Corpus of Present-Day American English (*or simply* Brown Corpus)

**Dataset Provided By:** NLTK (Natural Language Tool Kit) Platform, by the NLTK Project, University of Pennsylvania

**Dataset Accessible Through:** Kaggle Datasets at [https://www.kaggle.com/datasets/nltkdata/brown-corpus](https://www.kaggle.com/datasets/nltkdata/brown-corpus)

**Dataset License/Usability:** As per the NLTK dataset licensing information ([https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md)),
the Brown Corpus can be used for educational purposes, with usage being restricted to non-commercial purposes.
Confirmed through the linguistics group ICAME (International Computer Archive of Modern and Medieval English), who are the distributor of the corpus,
that academic use is permitted ([https://icame.info/icame-corpora/](https://icame.info/icame-corpora/)).

**Key Data Risks:** Primary data risks include incorrectly tagged data and insufficient data for the training stage (including joint training of word embeddings).
Incorrectly tagged data should be minimal as the dataset was reviewed by humans and has been in use for decades, but there is still some risk/uncertainty.

**Dataset Split:** The Brown Corpus is a single collection, meaning a custom train/validation/test dataset split will be required.

### Evaluation Metrics

**Key Evaluation Metric:** Accuracy on the test split of the dataset - accuracy defined as percentage of words tagged correctly by the model.

**Supplementary Evaluation Metrics:** Zero-shot human-generated example accuracy - as the models will be tasked with language processing
and the author of this portfolio is a human who speaks the natural language in question, accuracy on a small number of author-created
and author-tagged sentences can be used for demonstration/supplementary purposes.

