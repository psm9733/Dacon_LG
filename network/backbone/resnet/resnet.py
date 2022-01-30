import tensorflow as tf
from network.common.blocks import StemBlock
from network.backbone.resnet.body import ResNetBody
from utils.utils import get_flops

def ResNet(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage, activation,
               weight_decay):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    body_out = ResNetBody(in_tensor, n_block_per_stage, filter_per_stage,
               kernel_size_per_stage, strides_per_stage, activation, weight_decay)
    return body_out


if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (64, 64, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], activation,
                      weight_decay)
    resnet50 = ResNet(stem, [3, 4, 6, 3],
                                 [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]],
                                 [(3, 3), (3, 3), (3, 3), (3, 3)], [(2, 2), (2, 2), (2, 2), (2, 2)], "relu", weight_decay)
    model = tf.keras.Model(inputs=[input_tensor], outputs=resnet50)
    model.summary()
    get_flops(model)
    # model.save("ResNet.h5")