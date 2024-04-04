import numpy as np
from sklearn.model_selection import train_test_split
import utils
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

file_path = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/'
data, target = np.load(file_path+'sarcasm-v1.npy', allow_pickle=True)

data_train, data_dev, target_train, target_dev = train_test_split(data, target, test_size=0.3, random_state=42)

train_tups = (data_train, target_train)
dev_tups = (data_dev, target_dev)

def word_feats(words) -> dict:
    return dict([(word, True) for word in words])

train_data = [(word_feats(words), label) for words, label in zip(train_tups[0], train_tups[1])]
dev_data = [(word_feats(words), label) for words, label in zip(dev_tups[0], dev_tups[1])]

classifier = NaiveBayesClassifier.train(train_data)

dev_y = [label for words, label in dev_data]
preds = [classifier.classify(word_feats(words)) for words, label in dev_data]

(f1, accuracy) = utils.get_prfa(dev_y, preds)

print("F1 score: ", f1)
print("Accuracy: ", accuracy)