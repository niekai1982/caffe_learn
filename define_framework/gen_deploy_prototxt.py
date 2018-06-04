import sys
sys.path.insert(0, '/Users/niekai/caffe-ssd/caffe/python')

import caffe
from caffe import to_proto
from caffe import layers as L
from caffe import  params as P

# conv1_relu = mx.sym.Activation(conv1_batchnorm, act_type='relu', name='conv1_relu')
# conv2_dw =mx.sym.Convolution(conv1_relu, name='conv2_dw', num_filter=2, kernel=(3,3), pad=(1, 1), no_bias=True, num_group=2)
# conv2_dw_batchnorm = mx.sym.BatchNorm(conv2_dw, name='conv2_dw_batchnorm', fix_gamma=True, eps=1e-5, momentum=0.9, axis=1)
# conv2_dw_relu = mx.sym.Activation(conv2_dw_batchnorm, act_type='relu', name='conv2_dw_relu')
# conv2 = mx.sym.Convolution(conv2_dw_batchnorm, name='conv2', num_filter=2, kernel=(1,1), pad=(0, 0), no_bias=True, num_group=1)
# flatten = mx.sym.flatten(data=conv2)
# fc = mx.sym.FullyConnected(data=flatten, num_hidden=4)
# net = mx.sym.SoftmaxOutput(data=fc, name='softmax')


def sample_net():
    conv1 = L.Convolution(name='conv1', bottom='data', kernel_size=3, num_output=2, stride=1, pad=1, bias_term=False, group=1, weight_filler=dict(type='xavier'))
    conv1_batchnorm = L.BatchNorm(conv1, name='conv1_batchnorm', eps=1e-5, use_global_stats=1)
    conv1_relu = L.ReLU(conv1_batchnorm, name='con1_batchnorm_relu')
    conv2_dw = L.Convolution(conv1_relu, name='conv2_dw',kernel_size=3, num_output=2, stride=1, pad=1, bias_term=False, group=2, weight_filler=dict(type='xavier') )
    conv2_dw_batchnorm = L.BatchNorm(conv2_dw, name='conv2_dw_batchnorm', eps=1e-5, use_global_stats=1)
    #conv2_dw_relu = L.ReLU(conv2_dw_batchnorm, name='con2_batchnorm_relu')
    conv2 = L.Convolution(conv2_dw_batchnorm, name='conv2',kernel_size=1, num_output=2, stride=1, pad=0, bias_term=False, group=1, weight_filler=dict(type='xavier') )
    conv2_perm = L.Permute(conv2, name="conv2_perm", order=(1,0))
    flatten = L.Flatten(conv2,name='flatten', axis=1)
    fc = L.InnerProduct(flatten, name='fc', num_output=4, weight_filler=dict(type='xavier'))
    # prob = L.SoftMax(fc,name='softmax')
    return to_proto(fc)

def write_deploy(file_name):
    with open(file_name, 'w') as f:
        f.write('name:"test_net"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:20\n')
        f.write('input_dim:20\n')
        f.write(str(sample_net()))

if __name__ == '__main__':
    write_deploy("sample_net_deploy.prototxt")




