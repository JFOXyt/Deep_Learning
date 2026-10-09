import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import keyboard
import tkinter

INPUT_NODES = 784
HIDDENLAYER_1 = 64
HIDDENLAYER_2 = 32
HIDDENLAYER_3 = 16
OUTPUT_NODES = 10
LAYER_TO_CALC = 4

LEARNING_RATE = 0.001
LAMBDA = 0.0001
BATCH = 32

MNIST_T = 60000
MNIST_TE = 10000

train = True
a = 0
c = 0

inputt = np.array([])
label_value = np.array(np.zeros(10))

if os.path.exists("neural_network\weights1.npy"):
    weights_matriz1 = np.load("neural_network\weights1.npy")
    weights_matriz2 = np.load("neural_network\weights2.npy")
    weights_matriz3 = np.load("neural_network\weights3.npy")
    weights_matriz4 = np.load("neural_network\weights4.npy")
    weights_matriz = [weights_matriz1, weights_matriz2, weights_matriz3, weights_matriz4]

    bias_matriz1 = np.load(r"neural_network\bias1.npy")
    bias_matriz2 = np.load(r"neural_network\bias2.npy")
    bias_matriz3 = np.load(r"neural_network\bias3.npy")
    bias_matriz4 = np.load(r"neural_network\bias4.npy")
    bias_matriz = [bias_matriz1, bias_matriz2, bias_matriz3, bias_matriz4]

else:
    weights_matriz1 = np.random.randn(HIDDENLAYER_1, INPUT_NODES) * np.sqrt(2 / INPUT_NODES)
    weights_matriz2 = np.random.randn(HIDDENLAYER_2, HIDDENLAYER_1) * np.sqrt(2 / HIDDENLAYER_1)
    weights_matriz3 = np.random.randn(HIDDENLAYER_3, HIDDENLAYER_2) * np.sqrt(1 / HIDDENLAYER_2)
    weights_matriz4 = np.random.randn(OUTPUT_NODES, HIDDENLAYER_3) * np.sqrt(1 / HIDDENLAYER_3)
    weights_matriz = [weights_matriz1, weights_matriz2, weights_matriz3, weights_matriz4]

    bias_matriz1 = np.zeros(HIDDENLAYER_1)
    bias_matriz2 = np.zeros(HIDDENLAYER_2)
    bias_matriz3 = np.zeros(HIDDENLAYER_3)
    bias_matriz4 = np.zeros(OUTPUT_NODES)
    bias_matriz = [bias_matriz1, bias_matriz2, bias_matriz3, bias_matriz4]

weight_error_sum = [
    np.zeros_like(weights_matriz[0]),
    np.zeros_like(weights_matriz[1]),
    np.zeros_like(weights_matriz[2]),
    np.zeros_like(weights_matriz[3])
]

bias_error_sum = [
    np.zeros_like(bias_matriz[0]),
    np.zeros_like(bias_matriz[1]),
    np.zeros_like(bias_matriz[2]),
    np.zeros_like(bias_matriz[3])
]

X, y = fetch_openml(
        "mnist_784", version = 1, return_X_y = True, as_frame=False)

X = X.astype(np.float32) / 255.0
y = y.astype(int)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


def get_mnistdataset(train : bool, index : int):

    if train:
        return X_train[index], y_train[index]
    else:
        return X_test[index], y_test[index]


def ReLu(a):
    return np.maximum(0, a)


def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / np.sum(exp_a)


def labelvalue_r(label):
    label_value[:] = 0
    label_value[label] = 1
    return label_value
    
def calculate(a_m, w_m, b_m):

    activated_results = []

    for layer in range(LAYER_TO_CALC):
        result = (w_m[layer] @ a_m + b_m[layer])
        if layer < LAYER_TO_CALC - 1:            
            a = ReLu(result)
            activated_results.append(a)
        else:
            a = softmax(result)
            
        a_m = a

    return a_m, activated_results


def errorsfunction(par_m, label, activated_results):

    E4 = par_m - label
    E3 = (weights_matriz[3].T @ E4) * (activated_results[2] > 0)
    E2 = (weights_matriz[2].T @ E3) * (activated_results[1] > 0)
    E1 = (weights_matriz[1].T @ E2) * (activated_results[0] > 0)  

    return E1, E2, E3, E4


def avgg():
    for layer in range(LAYER_TO_CALC):
        weight_error_sum[layer] += weight_erors[layer]
        bias_error_sum[layer] += errors[layer]

    return weight_error_sum, bias_error_sum


def gradient(errors,activated_results, inputt):

    weight_l = []

    weight_l.append(np.outer(errors[0], inputt))
    weight_l.append(np.outer(errors[1], activated_results[0]))
    weight_l.append(np.outer(errors[2], activated_results[1]))
    weight_l.append(np.outer(errors[3], activated_results[2]))

    return weight_l

    
def update_bw(w_m, b_m):

    for layer in range(LAYER_TO_CALC):

        avg_w_g = weight_error_sum[layer] / BATCH
        avg_b_g = bias_error_sum[layer] / BATCH

        w_m[layer] -= LEARNING_RATE * (avg_w_g + LAMBDA * w_m[layer])
        b_m[layer] -= LEARNING_RATE * avg_b_g

        weight_error_sum[layer].fill(0)
        bias_error_sum[layer].fill(0)

if train :
    count = MNIST_T
else:
    count = MNIST_TE

cont = False

if train:

    plt.ion()
    fig, ax = plt.subplots()
    linetrain, = ax.plot([], [])
    linetest, = ax.plot([], [])

    epochs = []
    correct = np.array([])
    test_times = []
    test_acuracy = np.array([])

    for i in range(500):
        a = 0
        c = 0

        indices = np.random.permutation(MNIST_T)

        for image_num, image in enumerate(indices):
            
            data = get_mnistdataset(train, image)
            inputt = data[0]
            label = int(data[1])

            calc_data = calculate(inputt, weights_matriz, bias_matriz)

            R = calc_data[0]
            activated_results = calc_data[1]

            Rt = labelvalue_r(label)

            errors = errorsfunction(R, Rt, activated_results)
            weight_erors = gradient(errors, activated_results, inputt)

            avgl = avgg()
            weight_error_sum = avgl[0]
            bias_error_sum = avgl[1]

            if (image_num+1) % BATCH == 0:
                update_bw(weights_matriz, bias_matriz)

            if image_num % 1000 == 0:
                print(f"{image_num}: predicted {np.argmax(R)}, actual {label}")

            if np.argmax(R) == label:
                c+=1

            if keyboard.is_pressed("k"):
                cont = True
                break

        print(f"Epoch {i} : Got {c} right")

        indices = np.random.permutation(MNIST_TE)

        for image_num, image in enumerate(indices):

            if keyboard.is_pressed("k") or cont:
                break
    
            data = get_mnistdataset(False, image)
            inputt = data[0]
            label = int(data[1])
    
            calc_data = calculate(inputt, weights_matriz, bias_matriz)
            R = calc_data[0]
    
            if image_num % 1000 == 0:
                print(f"{image_num}: predicted {np.argmax(R)}, actual {label}")
            
            if np.argmax(R) == label:
                a+=1
    
        print(f"Test: Got {a} right")

        epochs.append(i)
        correct = np.append(correct, c)

        test_times.append(i)
        test_acuracy = np.append(test_acuracy, a)

        print(f"Epoch {i} : Got {c} right, Test {i}: Got {a} right")

        linetrain.set_data(epochs, correct*100/MNIST_T)
        linetest.set_data(test_times, test_acuracy*100/MNIST_TE)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        if keyboard.is_pressed("k") or cont:
            break

    plt.ioff()
    plt.show()

else:
    indices = np.random.permutation(MNIST_TE)
    wonglabel = []
    thinklabel = []
    wongimage = []

    for image_num, image in enumerate(indices):

        data = get_mnistdataset(train, image)
        inputt = data[0]
        label = int(data[1])

        calc_data = calculate(inputt, weights_matriz, bias_matriz)
        R = calc_data[0]

        if image_num % 1000 == 0:
            print(f"{image_num}: predicted {np.argmax(R)}, actual {label}")
        
        if np.argmax(R) == label:
            c+=1
        else:
            wongimage.append(inputt * 255)
            wonglabel.append(label)
            thinklabel.append(np.argmax(R))

    print(f"Got {c} right")

    root = tkinter.Tk()
    root.geometry(f"{28*15}x{28*15+75}")

    index = 0

    image_frame = tkinter.Frame(
        root,
        width=28*15,
        height=28*15
    )
    image_frame.place(x=0, y=0)

    label_l = tkinter.Label(
        root,
        font=("Arial", 20, "bold")
    )
    label_l.place(x=10, y=28*15)

    def createtk(index):

        for widget in image_frame.winfo_children():
            widget.destroy()

        for i, pixel in enumerate(wongimage[index]):

            x = (i % 28) * 15
            y = (i // 28) * 15

            pix = int(pixel)

            color = f"#{pix:02x}{pix:02x}{pix:02x}"

            frame = tkinter.Frame(
                image_frame,
                width=15,
                height=15,
                bg=color
            )

            frame.place(x=x, y=y)

        label_l.config(
            text=f"predicted {thinklabel[index]}, actual {wonglabel[index]}"
        )


    def next_image():
        global index

        index += 1

        if index >= len(wongimage):
            index = 0

        createtk(index)

    createtk(index)

    next_button = tkinter.Button(
        root,
        text="next",
        command=next_image
    )

    next_button.place(
        x=(28*15)-50,
        y=(28*15)+25
    )

    root.mainloop()


np.save("neural_network\weights1.npy", weights_matriz[0])
np.save("neural_network\weights2.npy", weights_matriz[1])
np.save("neural_network\weights3.npy", weights_matriz[2])
np.save("neural_network\weights4.npy", weights_matriz[3])

np.save(r"neural_network\bias1.npy", bias_matriz[0])
np.save(r"neural_network\bias2.npy", bias_matriz[1])
np.save(r"neural_network\bias3.npy", bias_matriz[2])
np.save(r"neural_network\bias4.npy", bias_matriz[3])

print("Saved weights and bias")

