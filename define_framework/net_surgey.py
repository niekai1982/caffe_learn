import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/Users/niekai/caffe-ssd/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')


import caffe

caffe.set_mode_cpu()

#load prototxt
net = caffe.Net('sample_net_deploy.prototxt', caffe.TEST)

data_input = np.zeros(shape=net.blobs['data'].data.shape)

conv1_params = np.ones(shape=net.params['conv1'][0].data.shape)

net.params['conv1'][0].data[...] = conv1_params

#define mean val
net.params['conv1_batchnorm'][0].data[0] = 1
net.params['conv1_batchnorm'][0].data[1] = 1

#define var val
net.params['conv1_batchnorm'][1].data[0] = 1
net.params['conv1_batchnorm'][1].data[1] = 1

#scale factor
# mean = mean/scale_factor;var = var/scale_factor
# out = (out - mean) / (var + eps)**0.5
net.params['conv1_batchnorm'][2].data[0] = 0.5

net.blobs['data'].data[...] = data_input

out = net.forward()

print out


