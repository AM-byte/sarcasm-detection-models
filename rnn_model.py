import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.layers import LSTM

# Loading the data
file_path = '/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/'
data, target = np.load(file_path+'sarcasm-v1.npy', allow_pickle=True)
data2, target2 = np.load(file_path+'sarcasm-v2.npy', allow_pickle=True)
w2v_model_v1 = KeyedVectors.load_word2vec_format(file_path+'sarcasm-v1-embeddings.txt', binary=False)
w2v_model_v2 = KeyedVectors.load_word2vec_format(file_path+'sarcasm-v2-embeddings.txt', binary=False)

# Model creation
def create_model(word_index, max_length, embedding_matrix):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 50, input_length=max_length, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# Model trained on dataset 1
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index)+1, 50))

for word, i in word_index.items():
    embedding_vector = w2v_model_v1[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_length = max(len(x) for x in sequences)
X = pad_sequences(sequences, maxlen=max_length)
y = target.astype('float32')

model = create_model(word_index, max_length, embedding_matrix)
model.summary()

model.fit(X, y, epochs=15, batch_size=64, validation_split=0.2)

# # Model trained on dataset 2
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(data2)
sequences2 = tokenizer2.texts_to_sequences(data2)
word_index2 = tokenizer2.word_index

embedding_matrix2 = np.zeros((len(word_index2)+1, 50))

for word, i in word_index2.items():
    embedding_vector2 = w2v_model_v2[word]
    if embedding_vector2 is not None:
        embedding_matrix2[i] = embedding_vector2

X2 = pad_sequences(sequences2, maxlen=max_length)
y2 = target2.astype('float32')

model2 = create_model(word_index2, max_length, embedding_matrix2)
model2.summary()

model2.fit(X2, y2, epochs=15, batch_size=64, validation_split=0.2)
print('\n')
# Metrics
# Evaluating model 1
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Evaluating model 2
loss2, accuracy2 = model2.evaluate(X2, y2)
print(f'Loss: {loss2}, Accuracy: {accuracy2}')

# Evaluating model 1 performance on dataset 2
loss3, accuracy3 = model.evaluate(X2, y2)
print(f'Loss: {loss3}, Accuracy: {accuracy3}')

# Evaluating model 2 performance on dataset 1
loss4, accuracy4 = model2.evaluate(X, y)
print(f'Loss: {loss4}, Accuracy: {accuracy4}')


xlabels = ['Sarcasm-v1', 'Sarcasm-v2', 'v1 on v2', 'v2 on v1']
losses = [loss, loss2, loss3, loss4]
accuracies = [accuracy, accuracy2, accuracy3, accuracy4]

x = np.arange(len(xlabels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, losses, width, label='Loss')
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