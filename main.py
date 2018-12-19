import os
import random
import data_utils
import sift
import vgg16


if __name__ == '__main__':
    sift_data_file = './data/sift'
    if not os.path.exists(sift_data_file):
        sift.process_train_data(sift_data_file)
    train_des_list = sift.get_train_des_list(sift_data_file)

    vgg16_data_file = './data/vgg16'
    if not os.path.exists(vgg16_data_file):
        vgg16.process_train_data(vgg16_data_file)
    train_fea_list = vgg16.get_train_fea_list(vgg16_data_file)
    model = vgg16.VGGNet()

    convert_data = []
    for train_batch in data_utils.get_data():
        data = train_batch[b'data']
        for d in data:
            convert_data.append(data_utils.convert(d))

    num = 10
    for test_batch in data_utils.get_data(tag='test'):
        data = test_batch[b'data']
        sample_ids = random.sample(range(len(data)), 500)
        # Test
        total_sift_ssim, total_vgg16_ssim = 0.0, 0.0
        for sample_id in sample_ids:
            sift_ind = sift.image_retrieval(data[sample_id], train_des_list, num)
            sift_ssim = data_utils.eval(data[sample_id], sift_ind, convert_data)
            total_sift_ssim += sift_ssim
            vgg16_ind = vgg16.image_retrieval(data[sample_id], model, train_fea_list, num)
            vgg16_ssim = data_utils.eval(data[sample_id], vgg16_ind, convert_data)
            total_vgg16_ssim += vgg16_ssim
        print('SIFT Cos Sim: %s; VGG16 Cos Sim: %s' % (total_sift_ssim / len(sample_ids), total_vgg16_ssim / len(sample_ids)))
        # Case Study
        for i in range(5):
            sample_id = random.randint(0, len(data))
            sift_ind = sift.image_retrieval(data[sample_id], train_des_list, num)
            sift_ssim = data_utils.eval(data[sample_id], sift_ind, convert_data)
            vgg16_ind = vgg16.image_retrieval(data[sample_id], model, train_fea_list, num)
            vgg16_ssim = data_utils.eval(data[sample_id], vgg16_ind, convert_data)
            print('Sample %d, SIFT Cos Sim: %s; VGG16 Cos Sim: %s' % (i, sift_ssim, vgg16_ssim))
            data_utils.show_results(data[sample_id], sift_ind, vgg16_ind, convert_data, num)
