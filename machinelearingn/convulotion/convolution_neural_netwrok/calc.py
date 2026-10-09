import numpy as np
from scipy.ndimage import correlate
import skimage.measure
from func_utils import ReLU, softmax
from data import CONVOLOVE_SIZE, FILTERS, POOL_SIZE

def convolove(weights,bias,ap, index, active_res):

    list_map = []
    weights = np.squeeze(weights, axis = 1)

    for ind in range(FILTERS[index]):

        ap = np.squeeze(ap)
        ap = ap[:, np.newaxis, np.newaxis]

        mapa = correlate(ap, weights)

        mapa = mapa.reshape(CONVOLOVE_SIZE[index])

        mapa += bias[ind]

        mapa = ReLU(mapa)

        mapa = max_pooling(mapa)

        list_map.append(mapa)

    #print(np.asarray(list_map).shape)

    active_res.append(list_map)

    #print(np.asarray(active_res).shape)

    return np.array(list_map), active_res

def convolove_ch(weights, bias, ap, index, active_res):

    list_map = []

    #print(ap.shape)

    for ind in range(FILTERS[index]):
        temp_map = []

        for bia in range(FILTERS[index]):

            ap = np.squeeze(ap)
            
            mapa = correlate(ap, weights[ind])

            mapa = mapa.reshape(CONVOLOVE_SIZE[index])

            mapa += bia

            mapa = ReLU(mapa)

            mapa = max_pooling(mapa)

            temp_map.append(mapa)

        list_map.append(np.sum(temp_map, axis = 0))

    active_res.append(list_map)

    #print(np.asarray(active_res).shape)

    return np.array(list_map), active_res

def calcn(weights, bias, ap, active_res):

    a = np.ndarray.flatten(ap)
    weights = np.squeeze(weights, axis = (0, 1))

    a = (weights @ a + bias)
    a = softmax(a / 1000000)

    active_res.append(a)

    return np.array(a)

def max_pooling(maps):

    return skimage.measure.block_reduce(maps, POOL_SIZE, np.max)