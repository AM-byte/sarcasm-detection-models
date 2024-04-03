


# nltk for Naive Bayes and metrics
import nltk
import nltk.classify.util
from nltk.metrics.scores import (precision, recall, f_measure, accuracy)
from nltk.classify import NaiveBayesClassifier

# some potentially helpful data structures from collections
from collections import defaultdict, Counter

# so that we can make plots
import matplotlib.pyplot as plt



TRAIN_FILE = "movie_reviews_train.txt"
DEV_FILE = "movie_reviews_dev.txt"



# train_tups
# dev_tups



def word_feats(words, binarized_features=True) -> dict:    
    # taken from the lecture 7 notebook
    if binarized_features:
        return dict([(word, True) for word in words]) 
    else:
        return Counter(words)


# This function links every word feature with its respective label
def combine_data(reviews, labels, binarized_features=True):
    train_data = []
    for i in range(len(reviews)):
        tokens = reviews[i]
        label = labels[i]
        train_data.append((word_feats(tokens, binarized_features), label))
    
    return train_data
    
# Seperating the train tups and the dev tups.
all_tokenized_reviews_train = train_tups[0]
all_sentiments_train = train_tups[1] 

all_tokenized_reviews_dev = dev_tups[0]
all_sentiments_dev = dev_tups[1] 

# combine the data - package it in a format readable by the classifiers
training_data_bin = combine_data(all_tokenized_reviews_train, all_sentiments_train)
dev_data_bin = combine_data(all_tokenized_reviews_dev, all_sentiments_dev)

training_data_multi = combine_data(all_tokenized_reviews_train, all_sentiments_train,binarized_features = False)
dev_data_multi = combine_data(all_tokenized_reviews_dev, all_sentiments_dev, binarized_features=False)

# test to make sure that you can train the classifier and use it to classify a new example'
nb_classifier = NaiveBayesClassifier.train(training_data_bin)
nb_multi_classifier = NaiveBayesClassifier.train(training_data_multi)


# test to make sure that you can train the classifier and use it to classify a new example

# Take a new example from the dev data
new_example = dev_data_bin[0]
new_example_words = new_example[0]
new_example_label = new_example[1]
new_example_predicted_label = nb_classifier.classify(word_feats(new_example_words,binarized_features=True))

if new_example_label == new_example_predicted_label:
    print("Correctly classifies one new example")
    print("True label from the dev data:", new_example_label)
    print("Predicted label from classifier:",new_example_predicted_label)


dev_y = []
predicted_data = []
for words, labels in dev_data_bin:
    dev_y.append(labels)
    feats = word_feats(words)  # Assuming binarized features by default
    prediction = nb_classifier.classify(feats)
    predicted_data.append(prediction)

    
preds_multi = []
for words, label in dev_data_multi:
    transformed_words = word_feats(words, binarized_features=False)  # For count-based features
    prediction = nb_multi_classifier.classify(transformed_words)
    preds_multi.append(prediction)
        
# getting metrics from the sutils file for both the models
(precision, recall, f1, accuracy) = sutils.get_prfa(dev_y, predicted_data)
(precision_multi, recall_multi, f1_multi, accuracy_multi) = sutils.get_prfa(dev_y, preds_multi)

print("All the statistics for the binary model:")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
print("Accuracy: ", accuracy)
print("\n")

print("All the statistics for the Multi model:")
print("Precision: ", precision_multi)
print("Recall: ", recall_multi)
print("F1 score: ", f1_multi)
print("Accuracy: ", accuracy_multi)
print("\n")
print("The best choice is the Binary Naive Bayes Classifier")



percentage_of_training_data = list(range(10, 101, 10))
t_precision = []
t_recall = []
t_f1 = []
t_accuracy = []

for percentage in percentage_of_training_data:
    number_of_elements = 16 * percentage # 1600 elements in the dataset
    classifier = NaiveBayesClassifier.train(training_data_bin[:number_of_elements])

    dev_y_2 = []
    predicted_data_2 = []
    for words, labels in dev_data_bin:
        dev_y_2.append(labels)
        feats = word_feats(words)  # Assuming binarized features by default
        prediction = classifier.classify(feats)
        predicted_data_2.append(prediction)
    
    # Unpacking all the relevant values
    (precision, recall, f1, accuracy) = sutils.get_prfa(dev_y_2, predicted_data_2)
    
    t_precision.append(precision)
    t_recall.append(recall)
    t_f1.append(f1)
    t_accuracy.append(accuracy)




