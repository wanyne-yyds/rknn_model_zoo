import os
from pathlib import Path

img_dir = Path('/mnt/e/Code/rknn/rknn_model_zoo/datasets/Lane_Dataset/imgs')

txtfile = os.path.split(img_dir)[0] + '/datasets.txt'
f = open(txtfile, 'w')
for imgfile in img_dir.rglob('*.*g'):
    f.write(str(imgfile) + '\n')
f.close()