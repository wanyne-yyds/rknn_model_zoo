#!/bin/bash

# 用法：python convert.py model path [rv1103 | rv1106] [i8/fp] [output_path]

# python /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov5/python/convert.py \
#     /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov5/model/yolov5s_relu.onnx \
#     rv1103 \
#     i8 \
#     /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov5/model/yolov5s_relu.rknn

python /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov8_obb/python/convert.py \
    /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov8_obb/model/yolov8n-obb.onnx \
    rv1103 \
    i8 \
    /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov8_obb/model/yolov8n-obb.rknn

# python /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov8_seg/python/convert.py \
#     /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov8_seg/model/yolov8n-seg.onnx \
#     rv1103 \
#     i8 \
#     /mnt/e/Code/rknn/rknn_model_zoo/examples/yolov8_seg/model/yolov8n-seg.rknn