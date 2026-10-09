import numpy as np

def get_mnistdataset(train : bool, index : int, X_train : list, y_train: list, X_test : list, y_test : list):

    if train:
        return X_train[index], y_train[index]
    else:
        return X_test[index], y_test[index]

def label_list(label : int):

    label_l = np.zeros(10)
    label_l[label] = 1

    return label_l


def ReLU(a):
    return np.maximum(0, a)

def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / np.sum(exp_a)