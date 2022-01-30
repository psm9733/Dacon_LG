import tensorflow as tf

def TimeSerisModel(in_tensor, input_shape, num_classes, activation, weight_decay):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor = tf.keras.layers.LSTM(input_shape[0], activation=activation, input_shape = input_shape, return_sequences=True, kernel_regularizer=kernel_regularizer)(in_tensor)
    out_tensor = tf.keras.layers.LSTM(128, activation=activation, return_sequences=True, kernel_regularizer=kernel_regularizer)(out_tensor)
    out_tensor = tf.keras.layers.LSTM(256, activation=activation, return_sequences=True, kernel_regularizer=kernel_regularizer)(out_tensor)
    out_tensor = tf.keras.layers.LSTM(256, activation=activation, return_sequences=True, kernel_regularizer=kernel_regularizer)(out_tensor)
    out_tensor = tf.keras.layers.LSTM(128, activation=activation, return_sequences=False, kernel_regularizer=kernel_regularizer)(out_tensor)
    out_tensor = tf.keras.layers.Dense(num_classes, activation="softmax")(out_tensor)
    model = tf.keras.Model(inputs=[in_tensor], outputs=out_tensor)
    return model

if __name__ == "__main__":
    weight_decay = 5e-4
    input_shape = (10, 1)
    input_tensor = tf.keras.layers.Input(input_shape)
    activation = tf.nn.leaky_relu
    model = TimeSerisModel(input_tensor, input_shape, 4, activation, weight_decay)
    model.summary()

