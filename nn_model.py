import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import utils
from sklearn.metrics import f1_score, accuracy_score

# loading the data
file_path = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/'
data, target = np.load(file_path+'sarcasm-v1.npy', allow_pickle=True)

data_train, data_dev, target_train, target_dev = train_test_split(data, target, test_size=0.3, random_state=42)

train_tups = (data_train, target_train)
dev_tups = (data_dev, target_dev)

x_train, y_train, x_dev, y_dev, vocab = utils.data_feats(train_tups, dev_tups)
x_train = np.array(x_train.toarray())
y_train = np.array(y_train.astype(int))
x_dev = np.array(x_dev.toarray())
y_dev = np.array(y_dev.astype(int))


def create_model():
    model = Sequential()
    
    # input layer
    model.add(Dense(units=128, activation='relu', input_dim=x_train.shape[1]))
    # hidden layer
    model.add(Dense(units=64, activation='relu'))
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

preds = model.predict(x_dev)
preds = (preds > 0.5).astype(int) # making a classification decision based on predictions
preds_flatten = [pred[0] for pred in preds]

score = f1_score(y_dev, preds_flatten)
print("F1 Score:", score)

dev_loss, dev_accuracy = model.evaluate(x_dev, y_dev, verbose=1)
print(f'Dev Loss: {dev_loss}')
print(f'Dev Accuracy: {dev_accuracy}')