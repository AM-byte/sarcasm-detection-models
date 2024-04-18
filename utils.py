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
        print("F1 score: ", f1)
        print("Accuracy: ", accuracy)

    return (f1, accuracy)

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

def vectorize_sentences(sentences, word_vectors, vector_size=50):
    """ Vectorize sentences by averaging word vectors. """
    vectorized_data = np.zeros((len(sentences), vector_size))
    
    for i, sentence in enumerate(sentences):
        word_count = 0
        for word in sentence:
            if word in word_vectors:
                vectorized_data[i] += word_vectors[word]
                word_count += 1
        if word_count > 0:
            vectorized_data[i] /= word_count
    
    return vectorized_data

def data_feats(train_tups, dev_tups, word_vectors):
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []

    # Process training data
    train_text = train_tups[0]
    x_train = vectorize_sentences(train_text, word_vectors)
    y_train = train_tups[1]

    # Process development data
    dev_text = dev_tups[0]
    x_dev = vectorize_sentences(dev_text, word_vectors)
    y_dev = dev_tups[1]
    
    return x_train, y_train, x_dev, y_dev