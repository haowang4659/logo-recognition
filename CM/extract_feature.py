#!/usr/bin/env python
import h5py
import numpy as np
import sys
import random
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')

import caffe
import os
import copy

caffe.set_mode_cpu()

model_def = caffe_root + 'examples/VGG16-lstm/deploy_raw.prototxt'
#model_weights = caffe_root + 'models/Flickr_VGG16/flickr_vgg16_final.caffemodel'
model_weights = 'VGG16.v2.caffemodel'

net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

'''
image = caffe.io.load_image(caffe_root + 'data/FlickrLogos_crop/samples/adidas/160469021_1.jpg')
transformed_image = transformer.preprocess('data', image)
print image.shape, transformed_image.shape

net.blobs['data'].data[...] = transformed_image

output = net.forward()
output_prob = output['prob'][0]
print 'predicted class is: ', output_prob.argmax()


#for layer_name, blob in net.blobs.iteritems():
#    print layer_name + '\t' + str(blob.data.shape)

#for layer_name, param in net.params.iteritems():
#    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

print net.blobs['fc7'].data[0][:20]
#for val in net.blobs['fc7'].data[0]:
#    print val
'''
samples_root = caffe_root + 'data/FlickrLogos_crop/samples/'
classes = ['adidas', 'aldi', 'apple', 'becks', \
            'bmw', 'carlsberg', 'chimay', 'cocacola', \
            'corona', 'dhl', 'erdinger', 'esso', \
            'fedex', 'ferrari', 'ford', 'fosters', \
            'google', 'guiness', 'heineken', 'HP', \
            'milka', 'nvidia', 'paulaner', 'pepsi', \
            'rittersport', 'shell', 'singha', 'starbucks', \
            'stellaartois', 'texaco', 'tsingtao', 'ups']

class_index = dict(zip(range(len(classes)), classes))
class_map = dict(zip(classes, range(len(classes))))
random.shuffle(classes)
#print class_index
num_steps = 10
feature_len = 4096
batch_size = 1
num_respective = 5
class_num = len(classes)
features = {}
for cla in classes:
    if not features.has_key(cla):
        features[cla] = []
    jpgs = os.listdir(samples_root + cla)
    for jpg in jpgs[:num_respective]:
        image = caffe.io.load_image(samples_root + cla + '/' + jpg)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'][0]
        features[cla].append(copy.deepcopy(net.blobs['fc7'].data[0]))
        print 'image name:', jpg, 'oughta be: ', cla, 'predicted class is: ', class_index[output_prob.argmax()], 'score is: ',output_prob[output_prob.argmax()]


feats = np.zeros((class_num * num_respective, batch_size, feature_len))
labels = np.zeros((class_num * num_respective, batch_size, class_num))
cont_stream = np.ones((num_steps, batch_size))
target_output = np.zeros((class_num * num_respective, batch_size))
num = 0
for cla in classes:
    for fea in features[cla]:
        for size in range(batch_size):
            feats[num, size, :] = fea
            labels[num, size, class_map[cla]] = 1
            cont_stream[0, size] = 0
            target_output[num, size] = class_map[cla]
        num += 1

with h5py.File('sample_features.hdf5', 'w') as f:
    print feats.shape, labels.shape, cont_stream.shape
    f.create_dataset('sample_features', data=feats)
    f.create_dataset('sample_labels', data=labels)
    f.create_dataset('cont_stream', data=cont_stream)

with open('hdf5_chunk.txt', 'w') as f:
    f.write('./examples/VGG16-lstm/sample_features.hdf5')

