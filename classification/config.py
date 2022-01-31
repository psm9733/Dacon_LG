WEIGHT_DECAY = 1e-5
num_classes_list = [5, 6, 21, 4, 25]
LR_DECAY_LATE = 0.1
INPUT_SHAPE = (352, 352, 3)
LR = 5e-4
EPS = 1e-7
WARMUP_EPOCHS = 10
EPOCHS = 400
BATCH_SIZE = 64
N_CYCLES = 4
TRAIN_PATH = "C:/Users/sangmin/Desktop/Dacon_LG/dataset/data/train"
VALID_PATH = "C:/Users/sangmin/Desktop/Dacon_LG/dataset/data/train"
img_format = ".jpg"
save_dir = "./saved_models/"

disease_dict = {
    '00': 0,
    'a1': 1,
    'a2': 2,
    'a3': 3,
    'a4': 4,
    'a5': 5,
    'a6': 6,
    'a7': 7,
    'a8': 8,
    'a9': 9,
    'a10': 10,
    'a11': 11,
    'a12': 12,
    'b1': 13,
    'b2': 14,
    'b3': 15,
    'b4': 16,
    'b5': 17,
    'b6': 18,
    'b7': 19,
    'b8': 20
}

label_description = {
    "1_00_0" : 0,
    "2_00_0" : 1,
    "2_a5_2" : 2,
    "3_00_0" : 3,
    "3_a9_1" : 4,
    "3_a9_2" : 5,
    "3_a9_3" : 6,
    "3_b3_1" : 7,
    "3_b6_1" : 8,
    "3_b7_1" : 9,
    "3_b8_1" : 10,
    "4_00_0" : 11,
    "5_00_0" : 12,
    "5_a7_2" : 13,
    "5_b6_1" : 14,
    "5_b7_1" : 15,
    "5_b8_1" : 16,
    "6_00_0" : 17,
    "6_a11_1" : 18,
    "6_a11_2" : 19,
    "6_a12_1" : 20,
    "6_a12_2" : 21,
    "6_b4_1" : 22,
    "6_b4_3" : 23,
    "6_b5_1" : 24
}
