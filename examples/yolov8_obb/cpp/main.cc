// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <fstream>

#include "yolov8_obb.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.cpp"
#endif

void get_files_in_directory(const char *directory_path, std::vector<std::string> &files) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory_path)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if(strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
                char image_path[512];
                sprintf(image_path, "%s/%s", directory_path, ent->d_name);
                files.push_back(image_path);
            }
        }
        closedir(dir);
    } else {
        /* could not open directory */
        perror("");
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{

    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    // 创建输出文件
    std::string dir = "./output";
    std::string command = "mkdir -p " + dir;
    std::system(command.c_str());

    int ret;
    std::vector<std::string> files;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();
    
    auto start_init = std::chrono::steady_clock::now();
    ret = init_yolov8_obb_model(model_path, &rknn_app_ctx);
    auto end_init = std::chrono::steady_clock::now();
    auto duration_init = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init).count();
    std::cout << "Init 2 use: " << duration_init << " milliseconds" << std::endl;

    object_detect_result_list od_results;
    char text[256];

    if (ret != 0)
    {
        printf("init_yolov8_obb_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    get_files_in_directory(image_path, files);
    for (int i = 0; i < files.size(); i++) {
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));

        std::string fileName;
        std::string filePath = files[i].c_str();
        size_t pos = filePath.find_last_of("/\\");
        if (pos != std::string::npos)
            fileName = filePath.substr(pos + 1);
        else
            fileName = filePath;
        pos = fileName.find_last_of(".");
        if (pos != std::string::npos)
            fileName = fileName.substr(0, pos);
        
        std::string outImgName = fileName + ".png";
        std::string outTxtName = fileName + ".txt";
        std::string outImgPathStr = dir + "/" + outImgName;
        std::string outTxtPathStr = dir + "/" + outTxtName;
        const char *outImgPath = outImgPathStr.c_str();

        std::ofstream outTxtFile(outTxtPathStr.c_str());
        std::cout << "Processing " << files[i].c_str() << std::endl;
        ret = read_image(files[i].c_str(), &src_image);
    #if defined(RV1106_1103) 
        //RV1106 rga requires that input and output bufs are memory allocated by dma
        ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                           (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
        memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
        dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
        free(src_image.virt_addr);
        src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    #endif
        if (ret != 0)
        {
            printf("read image fail! ret=%d image_path=%s\n", ret, filePath);
            goto out;
        }
        ret = inference_yolov8_obb_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("init_yolov8_obb_model fail! ret=%d\n", ret);
            goto out;
        }
        // 画框和概率
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            int objCls = det_result->cls_id;
            float objProb = det_result->prob;

            std::string objName = coco_cls_to_name(objCls);
            std::cout << objName << " @ " << objProb << " (";
            for (int j = 0; j < 4; j++)
            {
                std::cout << det_result->ptsl[j].x << " " << det_result->ptsl[j].y << " ";
            }
            std::cout << ")" << std::endl;

            int points[4][2];
            for (int j = 0; j < 4; j++) {
                points[j][0] = det_result->ptsl[j].x;
                points[j][1] = det_result->ptsl[j].y;
            }

            outTxtFile << objName << " " << objProb << " " ;
            for (int j = 0; j < 4; j++) {
                outTxtFile << points[j][0] << " " << points[j][1];
                if (j != 3) {
                    outTxtFile << " ";
                }
            }
            outTxtFile << std::endl;

            draw_line(&src_image, points[0][0], points[0][1], points[1][0], points[1][1], COLOR_RED, 2);
            draw_line(&src_image, points[3][0], points[3][1], points[2][0], points[2][1], COLOR_RED, 2);
            draw_line(&src_image, points[0][0], points[0][1], points[3][0], points[3][1], COLOR_RED, 2);
            draw_line(&src_image, points[2][0], points[2][1], points[1][0], points[1][1], COLOR_RED, 2);

            sprintf(text, "%s %.1f%%", objName.c_str(), objProb * 100);
            draw_text(&src_image, text, points[3][0], points[3][1] - 20, COLOR_RED, 10);
        }
        write_image(outImgPath, &src_image);
        outTxtFile.close();
        
        if (src_image.virt_addr != NULL)
        {
        #if defined(RV1106_1103) 
            dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                    rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
        #else
            free(src_image.virt_addr);
        #endif
        }
    }
out:
    deinit_post_process();

    ret = release_yolov8_obb_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_obb_model fail! ret=%d\n", ret);
    }
    return 0;
}
