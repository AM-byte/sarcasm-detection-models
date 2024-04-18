import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import utils

file_path = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/'
data, target = np.load(file_path+'sarcasm-v1.npy', allow_pickle=True)

data_train, data_dev, target_train, target_dev = train_test_split(data, target, test_size=0.3, random_state=42)

train_tups = (data_train, target_train)
dev_tups = (data_dev, target_dev)

x_train, y_train, x_dev, y_dev, vocab = utils.data_feats(train_tups, dev_tups)

# Initialize and train classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Make predictions
y_pred = knn.predict(x_dev)

# Accuracy
print("Accuracy:", accuracy_score(y_dev, y_pred))
