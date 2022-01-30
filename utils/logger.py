import tensorflow as tf

class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_writer = tf.summary.create_file_writer(log_dir)
        self.log_writer.set_as_default()

    def write(self, name, data, step):
        tf.summary.scalar(name, data, step=step)