import numpy as np

def initialize_weights(KERNEL_SIZE, FILTERS):

    temp_l_w = []

    for index, size in enumerate(KERNEL_SIZE):
        for _ in range(FILTERS[index]):
            temp_l_w.append(np.random.randn(*size))

        if   index == 0:
            weights_c1 = np.array(temp_l_w)
        elif index == 1:
            weights_c2 = np.array(temp_l_w)
        elif index == 2:
            weights_out = np.array(temp_l_w)

        temp_l_w = []

    return weights_c1, weights_c2, weights_out


def initialize_bias(KERNEL_SIZE, FILTERS):

    temp_l_b = []

    for index, size in enumerate(KERNEL_SIZE):
        for _ in range(FILTERS[index]):
            if index == 2:
                temp_l_b.append(np.random.randn(size[1]))
                break

            temp_l_b.append(np.random.randn(size[0]))

        if   index == 0:
            bias_c1 = np.array(temp_l_b)
        elif index == 1:
            bias_c2 = np.array(temp_l_b)
        elif index == 2:
            bias_out = np.array(temp_l_b)

        temp_l_b = []

    return bias_c1, bias_c2, bias_out

#* print(initialize_bias([(1,5,5),(2,3,3),(1,10,100)], [2,4,1]))
#* print(initialize_weights([(1,5,5),(2,3,3),(1,10,100)], [2,4,1]))