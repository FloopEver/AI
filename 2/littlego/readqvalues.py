import os
import pickle
file = open("X_qvalues.txt", "rb")
a = pickle.load(file)
file.close()
file = open("O_qvalues.txt", "rb")
b = pickle.load(file)
file.close()
# file = open("history_status.txt", "rb")
# c = pickle.load(file)
# file.close()

print(a)