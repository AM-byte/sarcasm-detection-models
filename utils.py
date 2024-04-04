import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.

    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    vocab = []
    for sentence in all_train_data_X:
        for word in sentence:
            vocab.append(word)

    return list(set(vocab))

def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.

    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    precision = precision_score(dev_y, preds)
    recall = recall_score(dev_y, preds)
    f1 = f1_score(dev_y, preds)
    accuracy = accuracy_score(dev_y, preds)

    if verbose:
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1)
        print("Accuracy: ", accuracy)

    return (precision, recall, f1, accuracy)

def featurize(vocab: list, data_to_be_featurized_X: list) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    feature_map = np.zeros((len(data_to_be_featurized_X), len(vocab)), dtype=int)

    for i, sentence in enumerate(data_to_be_featurized_X):
        word_counts = Counter(sentence)
        for j, vocab_word in enumerate(vocab):
            if word_counts[vocab_word] > 0:
                feature_map[i,j] = 1
    
    return feature_map

def data_feats(train_tups, dev_tups):
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []

    vectorizer = CountVectorizer(binary=True)
    train_text = [' '.join(sentence) for sentence, label in zip(train_tups[0], train_tups[1])]
    x_train = vectorizer.fit_transform(train_text)
    y_train = train_tups[1]
    vocab = vectorizer.vocabulary_

    x_dev_text = [' '.join(sentence) for sentence, label in zip(dev_tups[0], dev_tups[1])]
    x_dev = vectorizer.transform(x_dev_text)
    y_dev = dev_tups[1]
    
    return x_train, y_train, x_dev, y_dev, vocab