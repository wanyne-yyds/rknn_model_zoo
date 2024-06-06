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

#include "yolov8_obb.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.cpp"
#endif

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

    int ret;
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

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);

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
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
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
        printf("%d @ (%d %d %d %d %d %d %d %d) %.3f\n", det_result->cls_id,
               det_result->ptsl[0].x, det_result->ptsl[0].y,
               det_result->ptsl[1].x, det_result->ptsl[1].y,
               det_result->ptsl[2].x, det_result->ptsl[2].y,
               det_result->ptsl[3].x, det_result->ptsl[3].y,
               det_result->prob);
       
        int rtx = det_result->ptsl[0].x;    // 右上角点
        int rty = det_result->ptsl[0].y;
        int rbx = det_result->ptsl[1].x;    // 右下角点
        int rby = det_result->ptsl[1].y;
        int ltx = det_result->ptsl[3].x;    // 左上角点
        int lty = det_result->ptsl[3].y;
        int lbx = det_result->ptsl[2].x;    // 左下角点
        int lby = det_result->ptsl[2].y;

        draw_line(&src_image, rtx, rty, rbx, rby, COLOR_RED, 2);
        draw_line(&src_image, ltx, lty, lbx, lby, COLOR_RED, 2);
        draw_line(&src_image, rtx, rty, ltx, lty, COLOR_RED, 2);
        draw_line(&src_image, lbx, lby, rbx, rby, COLOR_RED, 2);
        // draw_circle(&src_image, rtx, rty, 2, COLOR_RED, 5);
        // draw_circle(&src_image, rbx, rby, 2, COLOR_GREEN, 5);
        // draw_circle(&src_image, ltx, lty, 2, COLOR_BLUE, 5);
        // draw_circle(&src_image, lbx, lby, 2, COLOR_YELLOW, 5);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prob * 100);
        draw_text(&src_image, text, ltx, lty - 20, COLOR_RED, 10);
    }
    write_image("out.png", &src_image);
out:
    deinit_post_process();

    ret = release_yolov8_obb_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_obb_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
#if defined(RV1106_1103) 
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
        free(src_image.virt_addr);
#endif
    }

    return 0;
}
