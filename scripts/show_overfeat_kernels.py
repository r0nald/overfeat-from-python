import numpy as np
import matplotlib.pyplot as plt
import math

from OverFeatModelLoad.load_tensors import load_tensors


def normalize_img(img):
    '''
        Scale and shift img so that all elements are between
        [0.0; 1.0]
    '''
    return (img - np.amin(img))/np.amax(img - np.amin(img))


def main():
    # Valid values are 0 and 1
    overfeat_net_nr = 0
    tensors = load_tensors('OverFeat/data/default',
                           overfeat_net_nr)

    subplot_cols = 16
    subplot_rows = math.ceil(tensors[0].shape[0]/subplot_cols)

    for i in range(tensors[0].shape[0]):
        plt.subplot(subplot_rows, subplot_cols, i)
        # The first dimension of the 3-dimensional tensor is color channel.
        # Swap axes to make color the last dimension.
        img = tensors[0][i].swapaxes(0, 2)
        img = normalize_img(img)
        plt.imshow(img, interpolation='none')
    plt.show()

if __name__ == '__main__':
    main()
