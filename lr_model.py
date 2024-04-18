import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import utils
import warnings
warnings.filterwarnings('ignore')
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

file_path = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/'
data, target = np.load(file_path+'sarcasm-v1.npy', allow_pickle=True)

data_train, data_dev, target_train, target_dev = train_test_split(data, target, test_size=0.3, random_state=42)

train_tups = (data_train, target_train)
dev_tups = (data_dev, target_dev)

data2, target2 = np.load(file_path+'sarcasm-v2.npy', allow_pickle=True)

data2_train, data2_dev, target2_train, target2_dev = train_test_split(data2, target2, test_size=0.3, random_state=42)

train_tups2 = (data2_train, target2_train)
dev_tups2 = (data2_dev, target2_dev)

w2v_model_v1 = KeyedVectors.load_word2vec_format(file_path+'sarcasm-v1-embeddings.txt', binary=False)
w2v_model_v2 = KeyedVectors.load_word2vec_format(file_path+'sarcasm-v2-embeddings.txt', binary=False)

def train_and_print(x_train, y_train, x_dev, y_dev):
    y_train = y_train.astype(int)
    y_dev = y_dev.astype(int)
    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, y_train)
    preds = logistic_model.predict(x_dev)
    (f1, accuracy) = utils.get_prfa(y_dev, preds)

    print("F1 score: ", f1)
    print("Accuracy: ", accuracy)
    return f1, accuracy

print("Train (sarcasm-v1), Dev (sarcasm-v1)")
x_train, y_train, x_dev, y_dev = utils.data_feats(train_tups, dev_tups, w2v_model_v1)
f1_1, acc_1 = train_and_print(x_train, y_train, x_dev, y_dev)

print("Train (sarcasm-v2), Dev (sarcasm-v2)")
x_train, y_train, x_dev, y_dev = utils.data_feats(train_tups2, dev_tups2, w2v_model_v2)
f1_2, acc_2 = train_and_print(x_train, y_train, x_dev, y_dev)

print("Train (sarcasm-v1), Dev (sarcasm-v2)")
x_train, y_train, x_dev, y_dev = utils.data_feats(train_tups, dev_tups2, w2v_model_v1)
f1_3, acc_3 = train_and_print(x_train, y_train, x_dev, y_dev)

print("Train (sarcasm-v2), Dev (sarcasm-v1)")
x_train, y_train, x_dev, y_dev = utils.data_feats(train_tups2, dev_tups, w2v_model_v2)
f1_4, acc_4 = train_and_print(x_train, y_train, x_dev, y_dev)

xlabels = ['Sarcasm-v1', 'Sarcasm-v2', 'v1 on v2', 'v2 on v1']
f1_scores = [f1_1, f1_2, f1_3, f1_4]
accuracies = [acc_1, acc_2, acc_3, acc_4]

x = np.arange(len(xlabels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score')
rects2 = ax.bar(x + width/2, accuracies, width, label='Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by dataset and metric')
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

# Attach a text label above each bar in *rects*, displaying its height.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()