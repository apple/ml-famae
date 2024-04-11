import os
import torch
import numpy as np

data_dir = r"your_data_files"
output_dir = r"your_output_files"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files])
files.sort()

edf20_permutation = permutation = np.array([0, 3, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 1, 2, 4, 13, 5, 7, 12, 15])
files = files[edf20_permutation]

len_train = int(len(files) * 0.6)
len_valid = int(len(files) * 0.2)

######## TRAINing files ##########
training_files = files[:len_train]
# load files
X_train = np.load(training_files[0])["x"]
y_train = np.load(training_files[0])["y"]

for np_file in training_files[1:]:
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)
torch.save(data_save, os.path.join(output_dir, "train.pt"))

######## Validation ##########
validation_files = files[len_train:(len_train + len_valid)]
# load files
X_train = np.load(validation_files[0])["x"]
y_train = np.load(validation_files[0])["y"]

for np_file in validation_files[1:]:
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)
torch.save(data_save, os.path.join(output_dir, "val.pt"))

######## TesT ##########
test_files = files[(len_train + len_valid):]
# load files
X_train = np.load(test_files[0])["x"]
y_train = np.load(test_files[0])["y"]

for np_file in test_files[1:]:
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)
torch.save(data_save, os.path.join(output_dir, "test.pt"))
