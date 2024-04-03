import numpy as np

sarcasm_v1 = np.load('/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/sarcasm-v1.npy', allow_pickle=True)
sarcasm_v2 = np.load('/Users/arnavmahale/Documents/Northeastern University/Spring 24/CS 4120/finalProject/sarcasm-detection-models/sarcasm-v2.npy', allow_pickle=True)

print(sarcasm_v1[0][0])
print(sarcasm_v1[1][0])

print(sarcasm_v2[0][0])
print(sarcasm_v2[1][0])