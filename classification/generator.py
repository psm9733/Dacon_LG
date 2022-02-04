import tensorflow as tf
import tqdm
import math
import numpy as np
import cv2
import glob
import json
import pandas as pd
import time
from datetime import datetime
import albumentations
from classification.config import *


class MultiTask_Generator(tf.keras.utils.Sequence):
    def __init__(self, dataset_info_path, batch_size, input_shape, num_classes_list, augs, is_train=True,
                 label_smooting=False, output_stride=[32, 16, 8]):
        self.csv_features = ['측정시각', '내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
                        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
        self.dataset_info_path = dataset_info_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes_list = num_classes_list
        self.augs = augs
        self.is_train = is_train
        self.output_stride = output_stride
        self.label_smooting = label_smooting
        self.data = self.get_dataset(dataset_info_path)
        self.indexes = None
        self.on_epoch_end()

    def get_dataset(self, dataset_info_path):
        img_list = glob.glob(dataset_info_path + "/**/*.jpg", recursive=True)
        new_img_list = []
        for img in img_list:
            img = img.replace("\\", "/")
            new_img_list.append(img)
        temp_csv = pd.read_csv(new_img_list[0].replace(".jpg", ".csv"))[self.csv_features[1:]]
        max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        for img_path in tqdm.tqdm(new_img_list[1:]):
            csv = img_path.replace(".jpg", ".csv")
            temp_csv = pd.read_csv(csv)[self.csv_features[1:]]
            temp_csv = temp_csv.replace('-', np.nan).dropna()
            if len(temp_csv) == 0:
                continue
            temp_csv = temp_csv.astype(float)
            temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
            max_arr = np.max([max_arr, temp_max], axis=0)
            min_arr = np.min([min_arr, temp_min], axis=0)

        # feature 별 최대값, 최솟값 dictionary 생성
        self.csv_feature_dict = {self.csv_features[i+1]: [min_arr[i], max_arr[i]] for i in range(len(self.csv_features[1:]))}
        return new_img_list

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
        return x, y

    def _gaussian_2d_ellipse(self, shape, cx, cy, w, h):
        alpha = 0.54
        sigma_x = alpha * w / 6
        sigma_y = alpha * h / 6
        vx = np.linspace(0, shape[0] - 1, shape[0])
        vy = np.linspace(0, shape[1] - 1, shape[1])
        x, y = np.meshgrid(vx, vy)

        heatmap = np.exp(-(((x - cx) ** 2) / (2 * sigma_x ** 2 + EPS) + ((y - cy) ** 2) / (2 * sigma_y ** 2 + EPS)))
        # heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
        return heatmap

    def get_gaussian_epllipse_heatmap(self, heatmap, cx, cy, box_w, box_h):
        h, w = heatmap.shape
        gaussian_map = self._gaussian_2d_ellipse((h, w), cx, cy, box_w, box_h)
        heatmap = np.maximum(heatmap, gaussian_map)
        heatmap[heatmap < EPS] = 0
        return heatmap

    def toTimestamp(self, time_info):
        time_info = time_info.replace('2021', '1970')
        return time.mktime(datetime.strptime(time_info, '%Y-%m-%d %H:%M:%S').timetuple())

    def __data_gen(self, data):
        cv2.setNumThreads(0)
        batch_img = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                             dtype=np.float32)
        # batch_seg_0 = np.zeros(shape=(self.batch_size, int(self.input_shape[0] / self.output_stride[0]), int(self.input_shape[1] / self.output_stride[0])), dtype=np.float32)
        # batch_seg_1 = np.zeros(shape=(self.batch_size, int(self.input_shape[0] / self.output_stride[1]), int(self.input_shape[1] / self.output_stride[1])), dtype=np.float32)
        # batch_seg_2 = np.zeros(shape=(self.batch_size, int(self.input_shape[0] / self.output_stride[2]), int(self.input_shape[1] / self.output_stride[2])), dtype=np.float32)
        batch_area = np.zeros(shape=(self.batch_size, self.num_classes_list[0]), dtype=np.float32)
        batch_crop = np.zeros(shape=(self.batch_size, self.num_classes_list[1]), dtype=np.float32)
        batch_disease = np.zeros(shape=(self.batch_size, self.num_classes_list[2]), dtype=np.float32)
        batch_risk = np.zeros(shape=(self.batch_size, self.num_classes_list[3]), dtype=np.float32)
        batch_embedding = np.zeros(shape=(self.batch_size, EMBEDDING_LENGTH), dtype=np.float32)
        batch_total = np.zeros(shape=(self.batch_size, self.num_classes_list[4]), dtype=np.float32)
        img_list = []
        area_list = []
        crop_list = []
        disease_list = []
        bbox_list = []
        risk_list = []
        total_list = []
        for img_path in data:
            img = cv2.imread(img_path)
            img_list.append(img)
            json_path = img_path.replace(img_format, ".json")
            with open(json_path, "r") as file:
                json_data = json.load(file)
                annotations = json_data['annotations']
                area = annotations['area']
                crop = annotations['crop']
                disease = annotations['disease']
                bboxes = annotations['part']
                risk = annotations['risk']
                total = str(crop) + "_" + disease + "_" + str(risk)
                area_list.append(area)
                crop_list.append(crop)
                disease_list.append(disease)
                bbox_list.append(bboxes)
                risk_list.append(risk)
                total_list.append(total)

        for i in range(len(data)):
            img = cv2.imread(data[i])
            # origin = img.copy()
            img_shape = img.shape
            # img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            img = self.augs(image=img)['image'] / 255.

            csv_path = data[i].replace(".jpg", ".csv")
            csv = pd.read_csv(csv_path)[self.csv_features]
            csv = csv.replace('-', 0)
            for col in csv.columns[1:]:
                csv[col] = csv[col].astype(float) - self.csv_feature_dict[col][0]
                csv[col] = csv[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            if len(csv) == 0:
                continue
            csv_max = csv.max().to_numpy()
            csv_min = csv.min().to_numpy()
            csv_max[0] = pd.to_datetime(csv_max[0]).month / 12
            csv_min[0] = pd.to_datetime(csv_min[0]).month / 12
            gt = np.concatenate((csv_min, csv_max))

            area = tf.keras.utils.to_categorical((area_list[i] - 1), num_classes=self.num_classes_list[0])
            crop = tf.keras.utils.to_categorical((crop_list[i] - 1), num_classes=self.num_classes_list[1])
            disease = tf.keras.utils.to_categorical(disease_dict.get(disease_list[i]),
                                                    num_classes=self.num_classes_list[2])
            # resize_shape = img.shape
            # w_ratio = resize_shape[1] / img_shape[1]
            # h_ratio = resize_shape[0] / img_shape[0]
            # for bboxes in bbox_list:
            #     for bbox in bboxes:
            #         w = int(bbox['w'] * w_ratio)
            #         h = int(bbox['h'] * h_ratio)
            #         cx = int((bbox['x'] + w / 2) * w_ratio)
            #         cy = int((bbox['y'] + h / 2) * h_ratio)
            #         # origin = cv2.rectangle(origin, (int(bbox['x']), int(bbox['y'])) , (int(bbox['x'] + bbox['w']), int(bbox['y'] + bbox['h'])), (0, 255, 0), 1)
            #         batch_seg_0[i] = self.get_gaussian_epllipse_heatmap(batch_seg_0[i],
            #                                                          int(cx / self.output_stride[0]),
            #                                                          int(cy / self.output_stride[0]),
            #                                                          int(w / self.output_stride[0]),
            #                                                          int(h / self.output_stride[0]))
            #         batch_seg_1[i] = self.get_gaussian_epllipse_heatmap(batch_seg_1[i],
            #                                                          int(cx / self.output_stride[1]),
            #                                                          int(cy / self.output_stride[1]),
            #                                                          int(w / self.output_stride[1]),
            #                                                          int(h / self.output_stride[1]))
            #         batch_seg_2[i] = self.get_gaussian_epllipse_heatmap(batch_seg_2[i],
            #                                                          int(cx / self.output_stride[2]),
            #                                                          int(cy / self.output_stride[2]),
            #                                                          int(w / self.output_stride[2]),
            #                                                          int(h / self.output_stride[2]))
            # cv2.imshow("batch_seg_0[i]", batch_seg_0[i])
            # cv2.imshow("batch_seg_1[i]", batch_seg_1[i])
            # cv2.imshow("batch_seg_2[i]", batch_seg_2[i])
            # cv2.imshow("origin", origin)
            # cv2.waitKey()
            risk = tf.keras.utils.to_categorical(risk_list[i], num_classes=self.num_classes_list[3])
            total = tf.keras.utils.to_categorical(label_description.get(total_list[i]),
                                                  num_classes=self.num_classes_list[4])
            batch_img[i] = img
            batch_area[i] = area
            batch_crop[i] = crop
            batch_disease[i] = disease
            batch_risk[i] = risk
            batch_embedding[i] = gt
            batch_total[i] = total
        return [batch_img, batch_embedding], [batch_area, batch_crop, batch_disease, batch_risk, batch_total]
        # return [batch_img, batch_embedding], [np.expand_dims(batch_seg_0, -1), np.expand_dims(batch_seg_1, -1), np.expand_dims(batch_seg_2, -1), batch_area, batch_crop, batch_disease, batch_risk,
        #                    batch_total]


if __name__ == "__main__":
    gen = MultiTask_Generator(dataset_info_path=TRAIN_PATH, batch_size=1, input_shape=(576, 576, 3),
                              num_classes_list=num_classes_list, augs=None, is_train=True)
    for i in tqdm.tqdm(range(gen.__len__())):
        gen.__getitem__(i)
