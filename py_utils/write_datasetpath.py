import os
from pathlib import Path

def write_datasetpath(dataset_path):
    f = open(dataset_path + "/datasets.txt", 'w')
    for img_file in Path(dataset_path).rglob('*.*g'):
        f.write(str(img_file) + '\n')
    f.close()

if __name__ == '__main__':
    dataset_path = '/mnt/d/code/rknn/rknn_model_zoo/datasets/MSL_Dataset'
    write_datasetpath(dataset_path)
    print('Dataset path is written to datasets.txt')