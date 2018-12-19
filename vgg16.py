import pickle
import data_utils
import numpy as np
from numpy import dot
from numpy.linalg import norm
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


class VGGNet:
    def __init__(self):
        self.input_shape = (32, 32, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights=self.weight,
                           input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           pooling=self.pooling, include_top=False)
        self.model.predict(np.zeros((1, 32, 32, 3)))

    def extract_feat(self, img):
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0] / norm(feat[0])
        return norm_feat


def get_image_similarity(f1, f2):
    return dot(f1, f2) / (norm(f1) * norm(f2))


def process_train_data(file):
    model = VGGNet()
    fea_list = []
    for train_batch in data_utils.get_data(tag='train'):
        data = train_batch[b'data']
        for d in data:
            fea = model.extract_feat(data_utils.convert(d))
            fea_list.append(fea)
    save_f = open(file, 'wb')
    pickle.dump(fea_list, save_f, protocol=pickle.HIGHEST_PROTOCOL)
    save_f.close()


def get_train_fea_list(file):
    read_f = open(file, 'rb')
    fea_list = pickle.load(read_f)
    read_f.close()
    return fea_list


def image_retrieval(sample, model, train_fea_list, num):
    sample_fea = model.extract_feat(data_utils.convert(sample))
    sims = []
    for train_fea in train_fea_list:
        sims.append(get_image_similarity(sample_fea, train_fea))
    sims = np.asarray(sims)
    ind = np.argpartition(sims, -num)[-num:]
    return ind
