import tensorflow as tf
from network.backbone.regnet.regnet import RegNetY
from network.common.layers import Conv2dBnAct
from network.common.blocks import StemBlock

def FPN(in_tensor_list, activation, weight_decay, mode="add"):
    out_tensor_list = []
    if in_tensor_list[0].shape[1] > in_tensor_list[-1].shape[1]:
        in_tensor_list.reverse()
    for index, n_in_tensor in enumerate(in_tensor_list):
        out_tensor = n_in_tensor
        if index > 0:
            up_out_tensor = tf.keras.layers.UpSampling2D((2, 2))(out_tensor_list[index - 1])
            b, w, h, c = out_tensor.shape
            up_out_tensor = Conv2dBnAct(up_out_tensor, c, (1, 1), (1, 1), activation=activation, weight_decay=weight_decay)
            identity_out_tensor = out_tensor
            if mode == "add":
                out_tensor = tf.keras.layers.Add()([up_out_tensor, identity_out_tensor])
            elif mode == "concat":
                out_tensor = tf.keras.layers.Concatenate()([up_out_tensor, identity_out_tensor])
        out_tensor_list.append(out_tensor)
    return out_tensor_list


if __name__ == "__main__":
    input_tensor = tf.keras.layers.Input((416, 416, 3))
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], "relu",
                         1e-5)
    backbone = RegNetY(
        stem,
        [1, 1, 1],
        [[256, 256, 256], [512, 512, 512], [1024, 1024, 1024]],
        [(3, 3), (3, 3), (3, 3)],
        [(2, 2), (2, 2), (2, 2)],
        [32, 32, 32],
        "relu",
        1e-5
    )
    fpn = FPN(backbone, "relu", 1e-5, "concat")
    model = tf.keras.Model(inputs=[input_tensor], outputs=fpn)
    # model.summary()
    model.save("test.h5")