import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

train_list = ['./data/data_batch_1', './data/data_batch_2', './data/data_batch_3', './data/data_batch_4',
              './data/data_batch_5', ]
test_list = ['./data/test_batch']


def convert(d):
    d = d.reshape(3, 32, 32)
    red = d[0].reshape(1024, 1)
    green = d[1].reshape(1024, 1)
    blue = d[2].reshape(1024, 1)

    pic = np.hstack((red, green, blue))
    pic_rgb = pic.reshape(32, 32, 3)
    return pic_rgb


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def get_data(tag='train'):
    file_list = train_list if tag == 'train' else test_list
    for f in file_list:
        yield unpickle(f)


def eval(sample, ind, convert_data):
    total = 0.0
    for i in ind:
        a = convert(sample).reshape(32 * 32 * 3).astype('int64')
        b = convert_data[i].reshape(32 * 32 * 3).astype('int64')
        similarity = dot(a, b) / (norm(a) * norm(b))
        total += similarity
    return total / len(ind)


def show_results(sample, sift_ind, vgg16_ind, convert_data, num):
    ind = [sift_ind, vgg16_ind]
    titles = ['SIFT', 'VGG16']

    fig = plt.figure()
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

    for i in range(len(ind)):
        inner = gridspec.GridSpecFromSubplotSpec(3, int(num / 2),
                                                 subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        ax = plt.Subplot(fig, inner[0])
        ax.set_title(titles[i])
        fig.add_subplot(ax)
        plt.imshow(convert(sample))
        plt.axis('off')
        pos = int(num / 2)
        for j in ind[i]:
            ax = plt.Subplot(fig, inner[pos])
            fig.add_subplot(ax)
            plt.imshow(convert_data[j])
            plt.axis('off')
            pos += 1

    plt.show()
