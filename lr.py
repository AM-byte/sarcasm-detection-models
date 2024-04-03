import numpy as np

sarcasm_v1 = np.load('finalProject/datasets/sarcasm-v1.npy', allow_pickle=True)
sarcasm_v2 = np.load('finalProject/datasets/sarcasm-v2.npy', allow_pickle=True)

print(sarcasm_v1[0][0])