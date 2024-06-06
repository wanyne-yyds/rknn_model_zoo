#!/bin/bash

# 用法：python yolov5.py --mdoel_path {rknn_model} --target {target_platform} --img_show
# 其中，如果带上 --img_show 参数，则会显示结果图片
# 注：这里以 RV1103 平台为例，如果是其他开发板，则需要修改命令中的平台类型
python /mnt/d/code/rknn/rknn_model_zoo/examples/yolov5/python/yolov5.py \
    --model_path /mnt/d/code/rknn/rknn_model_zoo/examples/yolov5/model/yolov5s_relu.rknn \
    --anchors /mnt/d/code/rknn/rknn_model_zoo/examples/yolov5/model/anchors_yolov5.txt \
    --img_show
    # --target rv1103 \
    # --device_id 0 \