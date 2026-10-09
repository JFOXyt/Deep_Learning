import sys
import numpy as np
from sklearn.datasets import fetch_openml
from collections import deque
from matplotlib import pyplot as plt

from data import *
from calc import *
from init_bw import *
from func_utils import *
from backprop import *

float_formatter = "{:.3f}".format
np.set_printoptions(threshold=sys.maxsize, formatter={'float_kind':float_formatter})

Train = True

X, y = fetch_openml(
        "mnist_784", version = 1, return_X_y = True, as_frame=False)

X = X.astype(np.float32)
y = y.astype(int)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

weights = [initialize_weights(KERNEL_SIZE, FILTERS)[0],initialize_weights(KERNEL_SIZE, FILTERS)[1],initialize_weights(KERNEL_SIZE, FILTERS)[2]]

bias = [initialize_bias(KERNEL_SIZE, FILTERS)[0], initialize_bias(KERNEL_SIZE, FILTERS)[1], initialize_bias(KERNEL_SIZE, FILTERS)[2]]

weights_sum, bias_sum = set_sum_bw(weights, bias)

c = 0

for count in range(50):
    activated_result = []
    for i,_ in enumerate(X_test):
        dataset = get_mnistdataset(Train, i, X_train, y_train, X_test, y_test)
        a = dataset[0]
        a = a[:, np.newaxis, np.newaxis]

        label = dataset[1]

        result, activated_result = convolove(weights[0], bias[0], a, 0, activated_result)
        print("result shape after conv/pool 1:", result.shape)


        result, activated_result = convolove_ch(weights[1], bias[1], result, 1, activated_result)
        print("result shape after conv/pool 2:", result.shape)

        #*
        """for r in result:
            plt.imshow(r)
            plt.show()"""#*

        res = calcn(weights[2], bias[2], result, activated_result)
        print("result shape after fully connected layer:", res.shape)

        guess = res


        if i % 100 == 0:
            print(f"I think its a {np.argmax(guess)}, with a {np.max(guess) * 100:.1f}% confidence, in reality its a {label}")

        if i % BATCH == 0:
            print(errorsfunc(guess, label, weights, activated_result))
            #grad = gradient(errors, activated_result)
            #avg_wb = avgg(grad, weights_sum, bias_sum, errors)
            #weights, bias = update_bw(weights, weights_sum, bias, bias_sum)

        if np.argmax(guess) == label:
            c +=1

    print(f"Epoch {count}: I got {c}/{len(X_test)} right!")

#TODO Implement BackPropagation into main code
#TODO Better up implementation of forwardpropagation on main code
