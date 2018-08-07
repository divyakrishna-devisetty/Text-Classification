import os
import numpy as np
import re
# Model parameters
TRAIN_DATA_PICKLE = 'pickle_files/train_data.pickle'
VOCAB_PICKLE = 'pickle_files/vocab_data.pickle'
SENTENCE_PER_REVIEW = 16
WORDS_PER_SENTENCE = 10
EMBEDDING_DIM = 300
FILTER_WIDTHS_SENT_CONV = np.array([3, 4, 5])
NUM_FILTERS_SENT_CONV = 100
FILTER_WIDTHS_DOC_CONV = np.array([3, 4, 5])
NUM_FILTERS_DOC_CONV = 100
NUM_CLASSES = 2
DROPOUT_KEEP_PROB = 0.5
L2_REG_LAMBDA = 0.0
BATCH_SIZE = 64
NUM_EPOCHS = 100
EVALUATE_EVERY = 100
CHECKPOINT_EVERY = 100
NUM_CHECKPOINTS = 5
LEARNING_RATE = 1e-3
"""
http://migsena.com/deceptive_spam_part_1/
"""
original_pos = 'data/positive_polarity/truthful_from_TripAdvisor/'
original_neg = 'data/negative_polarity/truthful_from_Web/'

deceptive_pos = 'data/positive_polarity/deceptive_from_MTurk/'
deceptive_neg = 'data/negative_polarity/deceptive_from_MTurk/'


def get_review_links():
    deceptive_reviews_list = []
    truthful_reviews_list = []
    for fold in os.listdir(original_pos):
        foldLink = os.path.join(original_pos, fold)
        if os.path.isdir(foldLink):
            for f in os.listdir(foldLink):
                fileLink = os.path.join(foldLink, f)
                truthful_reviews_list.append(fileLink)

    for fold in os.listdir(original_neg):
        foldLink = os.path.join(original_neg, fold)
        if os.path.isdir(foldLink):
            for f in os.listdir(foldLink):
                fileLink = os.path.join(foldLink, f)
                truthful_reviews_list.append(fileLink)

    for fold in os.listdir(deceptive_pos):
        foldLink = os.path.join(deceptive_pos, fold)
        if os.path.isdir(foldLink):
            for f in os.listdir(foldLink):
                fileLink = os.path.join(foldLink, f)
                deceptive_reviews_list.append(fileLink)

    for fold in os.listdir(deceptive_neg):
        foldLink = os.path.join(deceptive_neg, fold)
        if os.path.isdir(foldLink):
            for f in os.listdir(foldLink):
                fileLink = os.path.join(foldLink, f)
                deceptive_reviews_list.append(fileLink)
    return truthful_reviews_list, deceptive_reviews_list


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
