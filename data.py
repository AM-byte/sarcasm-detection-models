import json
import numpy as np
import nltk
nltk.download('punkt')

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