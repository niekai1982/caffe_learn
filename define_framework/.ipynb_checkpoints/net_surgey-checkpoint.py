import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx

caffe_root = '/Users/niekai/caffe-ssd/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')


import caffe

caffe.set_mode_cpu()
ctx = mx.cpu()

#mxnet load pretrained model
syms, args, auxs = mx.model.load_checkpoint('test_net', epoch=1)
mod = mx.mod.Module(symbol=syms, context=ctx)
mod.bind(data_shapes=[('data',(1,3,20,20))])
mod.set_params(arg_params=args, aux_params=auxs)

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


