import numpy as np
from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import utils
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# loading the data
file_path = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/'
data, target = np.load(file_path+'sarcasm-v1.npy', allow_pickle=True)

data_train, data_dev, target_train, target_dev = train_test_split(data, target, test_size=0.3, random_state=42)

train_tups = (data_train, target_train)
dev_tups = (data_dev, target_dev)

x_train, y_train, x_dev, y_dev, vocab = utils.data_feats(binarized=True, use_count_vectorizer=True, vocab=[], train_tups=train_tups, dev_tups=dev_tups)
x_train = np.array(x_train.toarray())
y_train = np.array(y_train)
x_dev = np.array(x_dev.toarray())
y_dev = np.array(y_dev)


def create_model():
    model = Sequential()
    
    # hidden layer
    model.add(Dense(units=128, activation='relu', input_dim=x_train.shape[1]))
    # output layer
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    return model

# create/compile your model in this cell
# Binary
model = create_model()
model.summary()

model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_dev, y_dev), verbose=1)
