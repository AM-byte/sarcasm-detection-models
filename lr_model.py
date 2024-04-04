import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import utils
import warnings
warnings.filterwarnings('ignore')

# loading the data
file_path = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/'
data, target = np.load(file_path+'sarcasm-v1.npy', allow_pickle=True)

data_train, data_dev, target_train, target_dev = train_test_split(data, target, test_size=0.3, random_state=42)

train_tups = (data_train, target_train)
dev_tups = (data_dev, target_dev)

def train_and_print(x_train, y_train, x_dev, y_dev):
    y_train = y_train.astype(int)
    y_dev = y_dev.astype(int)
    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, y_train)
    preds = logistic_model.predict(x_dev)
    (precision, recall, f1, accuracy) = utils.get_prfa(y_dev, preds)
    
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
    print("Accuracy: ", accuracy)

x_train, y_train, x_dev, y_dev, vocab = utils.data_feats(train_tups, dev_tups)
train_and_print(x_train, y_train, x_dev, y_dev)