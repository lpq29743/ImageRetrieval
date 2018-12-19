import cv2
import pickle
import numpy as np
import data_utils


def get_image_similarity(d1, d2):
    if d1 is None or d2 is None:
        return 0.0
    similarity = 0.0
    SIFT_RATIO = 0.8
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    if len(matches[0]) != 1:
        for m, n in matches:
            if m.distance < SIFT_RATIO * n.distance:
                similarity += 1.0

    # Custom normalization for better variance in the similarity matrix
    if similarity == len(matches):
        similarity = 1.0
    elif similarity > 1.0:
        similarity = 1.0 - 1.0 / similarity
    elif similarity == 1.0:
        similarity = 0.1
    else:
        similarity = 0.0

    return similarity


def process_train_data(file):
    sift_instance = cv2.xfeatures2d.SIFT_create()
    des_list = []
    for train_batch in data_utils.get_data(tag='train'):
        data = train_batch[b'data']
        for d in data:
            _, des = sift_instance.detectAndCompute(data_utils.convert(d), None)
            des_list.append(des)
    save_f = open(file, 'wb')
    pickle.dump(des_list, save_f, protocol=pickle.HIGHEST_PROTOCOL)
    save_f.close()


def get_train_des_list(file):
    read_f = open(file, 'rb')
    des_list = pickle.load(read_f)
    read_f.close()
    return des_list


def image_retrieval(sample, train_des_list, num):
    sift_instance = cv2.xfeatures2d.SIFT_create()
    _, sample_des = sift_instance.detectAndCompute(data_utils.convert(sample), None)
    sims = []
    for train_des in train_des_list:
        sims.append(get_image_similarity(sample_des, train_des))
    sims = np.asarray(sims)
    ind = np.argpartition(sims, -num)[-num:]
    return ind
