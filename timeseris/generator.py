import cv2
import tensorflow as tf
import glob
import tqdm
import numpy as np
import math
import natsort
import pandas as pd
import time
from datetime import datetime

class TimeSerisGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir_path, batch_size, input_shape, num_classes, is_train):
        self.csv_features = ['측정시각', '내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
                        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
        self.dataset_dir_path = dataset_dir_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train
        self.data = self.get_dataset(dataset_dir_path)
        self.index = None
        self.on_epoch_end()

    def get_dataset(self, dataset_dir_path):
        csv_list = []
        temp_list = glob.glob(dataset_dir_path + "/**/*.csv", recursive=True)
        temp_list = natsort.natsorted(temp_list)
        for csv in temp_list:
            csv = csv.replace("\\", "/")
            csv_list.append(csv)
        return csv_list

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        data = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data)
        # return x, y

    def toTimestamp(self, time_info):
        time_info = time_info.replace('2021', '1970')
        return time.mktime(datetime.strptime(time_info, '%Y-%m-%d %H:%M:%S').timetuple())

    def __data_gen(self, data):
        cv2.setNumThreads(0)
        batch_features = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1]))
        batch_gt = np.zeros(shape=(self.batch_size, self.num_classes))
        for csv in data:
            csv = pd.read_csv(csv)[self.csv_features]
            csv = csv.replace("-", np.nan).dropna()
            if len(csv) == 0:
                continue
            csv_max = csv.max().to_numpy()
            csv_min = csv.min().to_numpy()
            csv_max[0] = self.toTimestamp(csv_max[0])
            csv_min[0] = self.toTimestamp(csv_min[0])
            gt = np.array([csv_max, csv_min])
            print(gt.shape)

        return batch_features, batch_gt





if __name__ == "__main__":
    input_shape = (10, 1)
    num_classes = 4
    gen = TimeSerisGenerator("C:/Users/sangmin/Desktop/Dacon_LG/dataset/data/train/", 1, input_shape, num_classes, True)
    for i in tqdm.tqdm(range(gen.__len__())):
        gen.__getitem__(i)