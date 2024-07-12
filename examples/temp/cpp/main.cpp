#include "rknn_api.h"
#include <iostream>

int main()
{
    rknn_context ctx = 0;
    const char *model_path = "/userdata/rknn_yolov8_obb_demo/model/yolov8n-obb.rknn";
	std::cout << "model_path: " << model_path << std::endl;
    int ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    std::cout << "ret: " << ret << std::endl;
}