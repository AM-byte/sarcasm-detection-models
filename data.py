import json
import numpy as np

file_path_1 = 'finalProject/datasets/Sarcasm_Headlines_Dataset.json'
file_path_2 = 'finalProject/datasets/Sarcasm_Headlines_Dataset_v2.json'

def get_data(file_path):
    data = []
    target = []

    with open(file_path, 'r') as file:
        for line in file:
            article = json.loads(line)
            data.append(article['headline'])
            target.append(article['is_sarcastic'])

    return np.array([np.array(data), np.array(target)])

data_1 = get_data(file_path_1)
data_2 = get_data(file_path_2)

np.save('finalProject/datasets/sarcasm-v1.npy', data_1)
np.save('finalProject/datasets/sarcasm-v2.npy', data_2)