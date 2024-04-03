
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter
import time
import sentiment_utils as sutils


TRAIN_FILE = "movie_reviews_train.txt"
DEV_FILE = "movie_reviews_dev.txt"


# train_tups
# dev_tups




all_reviews_train = train_tups[0]
all_labels_train = train_tups[1]

vocab = sutils.create_index(all_reviews_train)


BINARIZED = True
USE_COUNT_VECTORIZER = False



start = time.time()

featurized_data = sutils.featurize(vocab, all_reviews_train)


end = time.time()
print("That took:", end - start, "seconds")
print("Vocabulary size: ", len(vocab))



count_v =  CountVectorizer()
joined_sentences = []
for review in all_reviews_train:
    sentence = ' '.join(review)    
    joined_sentences.append(sentence)

start = time.time()

countVectFeat = count_v.fit_transform(joined_sentences)

end = time.time()
print("That took:", end - start, "seconds")

print("Vocabulary size: ", len(count_v.vocabulary_))



zero_count = 0
l_vect = 0
for vector in featurized_data:
    # number of elements
    l_vect += len(vector)
    # how many zeros in each vector summed
    zero_count += Counter(vector)[0]

zero_count_vectorizer = 0
l_vect_2 = 0
for vector in countVectFeat.toarray():
    # number of elements
    l_vect_2 += len(vector)
    # number of zeros in each vector summed
    zero_count_vectorizer += Counter(vector)[0]

# Print out the average % of entries that are zeros in each vector in the vectorized training data ---------

# percentage of zeros in our vector
math = (zero_count / l_vect) * 100

# percentage of zeros in the countvectorizer
math_2 = (zero_count_vectorizer / l_vect_2) * 100

average = (math + math_2) / 2
print("The average % of entries that are zeros in each vector in the vectorized training data is",average)




import warnings
warnings.filterwarnings('ignore')

x_training_data = sutils.featurize(vocab, train_tups[0], binary=True)
x_dev_set = sutils.featurize(vocab, dev_tups[0], binary=True)

# the y data is just all the labels
y_training_data = train_tups[1]
y_dev_set = dev_tups[1]

# creating and fitting the logistic regression
binary_logistic_regression = LogisticRegression()
binary_logistic_regression.fit(x_training_data, y_training_data)

# predict the y data for the validation and testing data 
predicted_data = binary_logistic_regression.predict(x_dev_set)

# Unpacking all the relevant values
(precision, recall, f1, accuracy) = sutils.get_prfa(y_dev_set, predicted_data)

print("All the statistics for the binary model using our featurize method in sutils:")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
print("Accuracy: ", accuracy, "\n")


# the initial data that we created above 'joined sentences is the data to feed into the vectorizer'
vect = CountVectorizer(binary=True)

# use the same method as before to create the dev training set
joined_sentences_dev = []
for review in dev_tups[0]:
    sentence = ' '.join(review)    
    joined_sentences_dev.append(sentence)
    
    
x_training_data_vectorized = vect.fit_transform(joined_sentences)
x_dev_set_vectorized = vect.transform(joined_sentences_dev)

# the y data is here (labelling)
y_training_data_vectorized = train_tups[1]
y_dev_set_vectorized = dev_tups[1]

# fit the model to the new data 
binary_logistic_regression.fit(x_training_data_vectorized, y_training_data_vectorized)

# predict after the model has been trained on the new data
predicted_data_vectorized = binary_logistic_regression.predict(x_dev_set_vectorized)


# Unpacking all the relevant values
(precision_vectorized, recall_vectorized, f1_vectorized, accuracy_vectorized) = sutils.get_prfa(y_dev_set_vectorized, predicted_data_vectorized)

print("All the statistics for the binary vectorized model:")
print("Precision: ", precision_vectorized)
print("Recall: ", recall_vectorized)
print("F1 score: ", f1_vectorized)
print("Accuracy: ", accuracy_vectorized, "\n")

# --------------------------------- MULTINOMIAL -----------------------------------
# All the x data created using featurising
x_training_data_m = sutils.featurize(vocab, train_tups[0], binary=False)
x_dev_set_m = sutils.featurize(vocab, dev_tups[0], binary=False)

# the y data is just all the labels
y_training_data_m = train_tups[1]
y_dev_set_m = dev_tups[1]

# creating and fitting the logistic regression
binary_logistic_regression = LogisticRegression()
binary_logistic_regression.fit(x_training_data_m, y_training_data_m)

# predict the y data for the validation and testing data 
predicted_data_m = binary_logistic_regression.predict(x_dev_set_m)

# Unpacking all the relevant values
(precision, recall, f1, accuracy) = sutils.get_prfa(y_dev_set_m, predicted_data_m)

print("All the statistics for the Multinomial model using our featurize method in sutils:")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
print("Accuracy: ", accuracy, "\n")


# the initial data that we created above 'joined sentences is the data to feed into the vectorizer'
vect = CountVectorizer(binary=False)

# use the same method as before to create the dev training set
joined_sentences_dev = []
for review in dev_tups[0]:
    sentence = ' '.join(review)    
    joined_sentences_dev.append(sentence)
    
    
x_training_data_vectorized_m = vect.fit_transform(joined_sentences)
x_dev_set_vectorized_m = vect.transform(joined_sentences_dev)

# the y data is here (labelling)
y_training_data_vectorized_m = train_tups[1]
y_dev_set_vectorized_m = dev_tups[1]

# fit the model to the new data 
binary_logistic_regression.fit(x_training_data_vectorized_m, y_training_data_vectorized_m)

# predict after the model has been trained on the new data
predicted_data_vectorized_m = binary_logistic_regression.predict(x_dev_set_vectorized_m)


# Unpacking all the relevant values
(precision_vectorized, recall_vectorized, f1_vectorized, accuracy_vectorized) = sutils.get_prfa(y_dev_set_vectorized_m, predicted_data_vectorized_m)

print("All the statistics for the Multinomial vectorized model:")
print("Precision: ", precision_vectorized)
print("Recall: ", recall_vectorized)
print("F1 score: ", f1_vectorized)
print("Accuracy: ", accuracy_vectorized)
print("\n\n\n\n")
print("The best choice is the binary model that uses the count vectorizer")


percentage_of_training_data = list(range(10, 101, 10))
t_precision = []
t_recall = []
t_f1 = []
t_accuracy = []

for percentage in percentage_of_training_data:
    number_of_elements = 16 * percentage # 1600 elements in the dataset
    binary_logistic_regression.fit(x_training_data_vectorized[:number_of_elements],y_training_data_vectorized[:number_of_elements])
    
    predicted_data = binary_logistic_regression.predict(x_dev_set_vectorized)
    # Unpacking all the relevant values
    (precision, recall, f1, accuracy) = sutils.get_prfa(y_dev_set_vectorized, predicted_data)
    
    t_precision.append(precision)
    t_recall.append(recall)
    t_f1.append(f1)
    t_accuracy.append(accuracy)





