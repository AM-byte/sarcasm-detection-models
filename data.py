import json
import numpy as np
import nltk
nltk.download('punkt')
from gensim.models import Word2Vec

file_path_1 = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/datasets/Sarcasm_Headlines_Dataset.json'
file_path_2 = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/datasets/Sarcasm_Headlines_Dataset_v2.json'

def generate_tuples_from_file(file_path):
    """
    Generates data from file formated like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]
    
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of lists of tokens and a list of int labels
    """
    data = []
    target = []

    with open(file_path, 'r') as file:
        for line in file:
            article = json.loads(line)
            # get the data
            headline = article['headline']
            label = article['is_sarcastic']
            # tokenize the data
            tokens = nltk.word_tokenize(headline)
            # append the data
            data.append(tokens)
            target.append(label)

    return (np.array(data, dtype=object), np.array(target))

data_1 = generate_tuples_from_file(file_path_1)
data_2 = generate_tuples_from_file(file_path_2)

np.save('/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/sarcasm-v1.npy', data_1)
np.save('/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/sarcasm-v2.npy', data_2)

data_1_embeddings = Word2Vec(sentences=data_1[0], vector_size=50, window=5, min_count=1)
data_2_embeddings = Word2Vec(sentences=data_2[0], vector_size=50, window=5, min_count=1)

data_1_embeddings.wv.save_word2vec_format('/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/sarcasm-v1-embeddings.txt', binary=False)
data_2_embeddings.wv.save_word2vec_format('/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/sarcasm-v2-embeddings.txt', binary=False)


# data_train, data_dev, target_train, target_dev = train_test_split(data, target, test_size=0.3, random_state=42)

# train_tups = (data_train, target_train)
# dev_tups = (data_dev, target_dev)

# x_train, y_train, x_dev, y_dev, vocab = utils.data_feats(train_tups, dev_tups)
# x_train = np.array(x_train.toarray())
# y_train = np.array(y_train.astype(int))
# x_dev = np.array(x_dev.toarray())
# y_dev = np.array(y_dev.astype(int))


# data2, target2 = np.load(file_path+'sarcasm-v2.npy', allow_pickle=True)

# data2_train, data2_dev, target2_train, target2_dev = train_test_split(data2, target2, test_size=0.3, random_state=42)

# train_tups2 = (data2_train, target2_train)
# dev_tups2 = (data2_dev, target2_dev)

# x_train2, y_train2, x_dev2, y_dev2, vocab2 = utils.data_feats(train_tups2, dev_tups2)
# x_train2 = np.array(x_train2.toarray())
# y_train2 = np.array(y_train2.astype(int))
# x_dev2 = np.array(x_dev2.toarray())
# y_dev2 = np.array(y_dev2.astype(int))
