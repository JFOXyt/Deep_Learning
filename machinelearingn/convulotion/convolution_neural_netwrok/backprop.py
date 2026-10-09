import numpy as np
from math import floor
from data import *

def errorsfunc(result_layer : list, label_l, weights : list, active_res : list):       #TODO: Fix BackPropagation, 3rd time :(
                                                                                       #TODO: Create Relation Between Ouput Layer -> Conv1 -> Conv2 -> Input (issue is the pooling :/)

    E4 = result_layer - label_l

    print("E4 shape: ", E4.shape)


    weights[2] = np.squeeze(weights[2])

    E3 = []
    E3.append((E4 @ weights[2]).T * (np.asarray(active_res[2]).flatten() > 0))

    #E3 = np.reshape(E3, MPOOL_SIZE[0])

    print("E3 shape: ", np.asarray(E3).shape)

    E3 = np.squeeze(E3)

    E2 = []


    active_res[0] = np.array(active_res[0])
    active_res[0] = np.squeeze(active_res[0])

    for filters, _ in enumerate(weights[1]):    
        for channel, _ in enumerate(weights[1][filters]):
        
            for x in range(floor(len(E3[channel, filters]) / len(weights[1][filters][channel]))):

                for y in range(floor(len(E3[channel, filters]) / len(weights[1][filters][channel]))):
                    
                    E2.append((weights[1][filters][channel] @ E3[filters, x:x+3, y:y+3]) * (active_res[1][channel, x:x+3, y:y+3] > 0))


    print("E2 shape: ", np.asarray(E2).shape)
    #print(np.asarray(weights[0]).shape)
    #print(np.asarray(active_res[0][0:0+3, 0:0+3]).shape)

    E1 = []
    kern_n = 0
    counter = 0

    for filters, _ in enumerate(weights[0]):

        for channel, _ in enumerate(weights[0][filters]):

            for x in range(floor( len(active_res[0][0]) / len(weights[0][filters][channel]))):
                for y in range(floor( len(active_res[0][0]) / len(weights[0][filters][channel]))):

                    E1.append(weights[0][filters][channel] @ E2[kern_n] * (active_res[0][channel, x:x+3, y:y+3] > 0))

                    if counter == 3:
                        kern_n+=1
                        counter = 0

                    else:
                        counter +=1

    print("E1 shape: ", np.asarray(E1).shape)
    E1 = np.reshape(E1, (2, 28, 28))

    #print(np.asarray(E1).shape)


    return E3.shape, E2.shape

def gradient(errors : list, active_res : list, weights, Input):

    #Make gradient
    NotImplemented

def avgg(grad : list, weight_sum : list, bias_sum : list, errors : list):

    for layer, _ in enumerate(LAYERS):
        weight_sum[layer] += grad[layer] / BATCH
        bias_sum[layer] += errors[layer] / BATCH

    return weight_sum, bias_sum

def set_sum_bw(b : list, w : list):
    weight_sum = []
    bias_sum = []


    for iw in w:
        weight_sum.append(np.zeros_like(iw))


    for ib in b:
        bias_sum.append(np.zeros_like(ib))


    return weight_sum, bias_sum

def update_bw(weight : list, weights_sum : list, bias : list, bias_sum : list):

    for layer, _ in enumerate(LAYERS):

        weight[layer] += LR * weights_sum[layer]
        bias[layer] += LR * bias_sum[layer]

        weights_sum[layer].fill(0)
        bias_sum[layer].fill(0)

    return weight, bias