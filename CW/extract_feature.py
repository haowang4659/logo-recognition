#!/usr/bin/env python
import h5py
import numpy as np
import sys
import random
caffe_root = '/workspace/wanghao/git-repository/caffe-CN/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import os
import copy
import shutil
caffe.set_mode_gpu()
caffe.set_device(1)
project_root = '/workspace/wanghao/git-repository/project/flickr_32/CN/'
model_def = project_root + 'deploy.prototxt'

#model_weights = caffe_root + 'models/Flickr_VGG_M/flickr_vgg_m_final.caffemodel'
model_weights = project_root + 'final.caffemodel'
samples_root = '/workspace/wanghao/git-repository/original_data/FlickrLogos_crop/sample/'

#class_index = dict(zip(range(len(classes)), classes))
#class_map = dict(zip(classes, range(len(classes))))
class_num = 32

feature_len = 1024 #4096
batch_size = 1
#num_max_per_class = 10
num_max_per_class = 10

def get_transformer(net,mu):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1)) #data : (H' x W' x K) ndarray     caffe_in : (K x H x W)
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0)) #RGB to BGR
    return transformer

def get_error_list(image_path,class_list):
    error_list=[]
    transformer = get_transformer(net, mu)
    error_num=0
    for cla in class_list:

        jpgs = os.listdir(image_path + cla)
        for jpg in jpgs:
            image = caffe.io.load_image(image_path + cla + '/' + jpg)  # RGB
            transformed_image = transformer.preprocess('data', image)  # BGR
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            output_prob = output['prob'][0]

            if cla != class_index[output_prob.argmax()]:
                error_num+=1
                print error_num, ' image name:', jpg, 'oughta be: ', cla, 'predicted class is: ', class_index[
                    output_prob.argmax()], 'score is: ', output_prob[output_prob.argmax()]
                error_list.append(cla + ' ' + jpg)
    return error_list

def write_error_list(path,error_list):
    with open(path, 'w') as f:
        for tuple in error_list:
            f.writelines(tuple + '\n')

def read_error_list(path):
    error_list=[]
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            error_list.append(line.split('\n')[0])
            pass
    return error_list

def get_feature(net,mu,class_list,class_index):
    features = {}

    transformer = get_transformer(net,mu)
    error_num = 0
    for cla in class_list:
        if not features.has_key(cla):
            features[cla] = []
        jpgs = os.listdir(samples_root + cla)
        for jpg in jpgs:
            image = caffe.io.load_image(samples_root + cla + '/' + jpg) #RGB
            transformed_image = transformer.preprocess('data', image)   #BGR
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            output_prob = output['prob'][0]
            features[cla].append(copy.deepcopy(net.blobs['fc7'].data[0]))
            #if cla != class_index[output_prob.argmax()]:
            error_num += 1
            print error_num,' image name:', jpg, 'oughta be: ', cla, 'predicted class is: ', class_index[output_prob.argmax()], 'score is: ',output_prob[output_prob.argmax()]


    return features


def write_feature(features,class_list):
    class_map = dict(zip(class_list, range(len(class_list))))
    feats = np.zeros((class_num * num_max_per_class, batch_size, feature_len))
    labels = np.zeros((class_num * num_max_per_class, batch_size, class_num))
    num = 0

    for cla in class_list:
        for fea in features[cla]:
            for size in range(batch_size):
                feats[num, size, :] = fea
                labels[num, size, class_map[cla]] = 1
            num += 1
    with h5py.File(project_root + 'EF/sample_features.hdf5', 'w') as f:
        print feats.shape, labels.shape
        f.create_dataset('sample_features', data=feats)
        f.create_dataset('sample_labels', data=labels)

    with open(project_root + 'EF/hdf5_chunk.txt', 'w') as f:
        f.write(project_root + 'EF/sample_features.hdf5')





def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print path

def get_M_images(image_path,sample_path,class_list,error_list):

    image_list_per_class = [os.listdir(image_path + class_name) for class_name in class_list]
    dict_class_image = dict(zip(class_list, image_list_per_class))

    #num_per_class = [len(dict_class_image[class_name]) for class_name in class_list]
    #dict_class_num = dict(zip(class_list, num_per_class))

    # matrix M image
    #sample_image_list_per_class = [dict_class_image[class_name][:num_max_per_class] for class_name in class_list]
    #dict_class_image = dict(zip(class_list, sample_image_list_per_class))

    for class_name, image_list in dict_class_image.items():
        mkdir(sample_path + class_name)
        print len(image_list)
        for image in image_list:
            if class_name + ' '+image not in error_list:

                shutil.copyfile(image_path + '/' + class_name + '/' + image, sample_path + class_name + '/' + image)
            if len(os.listdir(sample_path + class_name))==num_max_per_class:
                break;


if __name__ == '__main__':

    image_path = '/workspace/wanghao/git-repository/original_data/FlickrLogos_crop/sample/'


    class_list=[]
    label_list=[]
    file = open('/workspace/wanghao/git-repository/original_data/FlickrLogos_crop/train.txt')

    while True:
        line = file.readline()
        if not line:
            break

        class_name = line.split('/')[0]
        class_label = int(line.split(' ')[1])
        if class_name not in class_list:
            class_list.append(class_name)
            label_list.append(class_label)
        pass

    class_index = dict(zip(label_list,class_list))



    net = caffe.Net(model_def, model_weights, caffe.TEST)
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')

    mu = mu.mean(1).mean(1)
    error_list=[]
    #error_list=get_error_list(image_path,class_list)
    #write_error_list(project_root + 'EF/error_list.txt',error_list)
    #error_list = read_error_list(project_root + 'EF/error_list.txt')
    #print error_list
    #print 'hello'

    #get_M_images(image_path,samples_root,class_list,error_list)
    features = get_feature(net,mu,class_list,class_index)
    write_feature(features,class_list)





    #while True:
    #    get_M_images(image_path, samples_root,error_list)
    #    featrues,error_list = get_feature(net,mu,class_list,class_index,error_list)
    #    if len(error_list)==0:
    #        break;
    #    print len(error_list)
    #print len(error_list)
    #write_feature(featrues,class_list,dict(zip(class_list,range(len(class_list)))))






