// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include "yolov8_obb.h"
#include <algorithm>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>
#define LABEL_NALE_TXT_PATH "./model/MSL_Labels.txt"

static char *labels[OBJ_CLASS_NUM];

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

void removeNewline(char* str) {
    while (*str) {
        if (*str == '\n' || *str == '\r') {
            *str = '\0';
        }
        str++;
    }
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        removeNewline(s);
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static float dot_2d(const Point& A, const Point& B) {
    return A.x * B.x + A.y * B.y;
}

static float cross_2d(const Point& A, const Point& B) {
    return A.x * B.y - B.x * A.y;
}

void rotatePoint(float x, float y, float cx, float cy, float angle, float& out_x, float& out_y) {
    float radians = (angle - 0.25) * 180.0;     // 将弧度转换为角度
    radians = (radians - 0.25) * M_PI / 180.0;  // 将角度转换为弧度
    float cos_angle = std::cos(radians);
    float sin_angle = std::sin(radians);
    // 对坐标进行旋转
    out_x = cos_angle * (x - cx) - sin_angle * (y - cy) + cx;
    out_y = sin_angle * (x - cx) + cos_angle * (y - cy) + cy;
}

static void get_rotated_vertices(const Object& box, Point(&pts)[4]) {
    float angle = box.angle;
    float xmin = box.xmin;
    float xmax = box.xmax;
    float ymin = box.ymin;
    float ymax = box.ymax;

    // 计算矩形的中心点坐标
    float cx = (xmin + xmax) / 2;
    float cy = (ymin + ymax) / 2;

    // 计算矩形的宽度和高度
    float width = xmax - xmin;
    float height = ymax - ymin;

    // 计算矩形的四个顶点坐标
    float half_width = width / 2;
    float half_height = height / 2;
    
    rotatePoint(cx - half_width, cy - half_height, cx, cy, angle, pts[0].x, pts[0].y); // 左上角顶点
    rotatePoint(cx + half_width, cy - half_height, cx, cy, angle, pts[1].x, pts[1].y); // 右上角顶点
    rotatePoint(cx + half_width, cy + half_height, cx, cy, angle, pts[2].x, pts[2].y); // 右下角顶点
    rotatePoint(cx - half_width, cy + half_height, cx, cy, angle, pts[3].x, pts[3].y); // 左下角顶点

}

static int get_intersection_points(const Point(&pts1)[4], const Point(&pts2)[4],
    Point(&intersections)[24])
{

    Point vec1[4], vec2[4];
    for (int i = 0; i < 4; i++) {
        vec1[i] = pts1[(i + 1) % 4] - pts1[i];
        vec2[i] = pts2[(i + 1) % 4] - pts2[i];
    }

    int num = 0;  // number of intersections
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float det = cross_2d(vec2[j], vec1[i]);
            if (fabs(det) <= 1e-14) {
                continue;
            }

            auto vec12 = pts2[j] - pts1[i];

            float t1 = cross_2d(vec2[j], vec12) / det;
            float t2 = cross_2d(vec1[i], vec12) / det;

            if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
                intersections[num++] = pts1[i] + vec1[i] * t1;
            }
        }
    }

    {
        const auto& AB = vec2[0];
        const auto& DA = vec2[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++) {
            auto AP = pts1[i] - pts2[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts1[i];
            }
        }
    }

    {
        const auto& AB = vec1[0];
        const auto& DA = vec1[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++) {
            auto AP = pts2[i] - pts1[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts2[i];
            }
        }
    }

    return num;
}

static int convex_hull_graham(const Point(&p)[24], const int& num_in, Point(&q)[24])
{
    int t = 0;
    for (int i = 1; i < num_in; i++) {
        if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
            t = i;
        }
    }
    auto& start = p[t];

    for (int i = 0; i < num_in; i++) {
        q[i] = p[i] - start;
    }

    auto tmp = q[0];
    q[0] = q[t];
    q[t] = tmp;

    float dist[24];
    for (int i = 0; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    std::sort(q + 1, q + num_in, [](const Point& A, const Point& B) -> bool {
        float temp = cross_2d(A, B);
        if (fabs(temp) < 1e-6) {
            return dot_2d(A, A) < dot_2d(B, B);
        }
        else {
            return temp > 0;
        }
        });

    for (int i = 0; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    int k;
    for (k = 1; k < num_in; k++) {
        if (dist[k] > 1e-8) {
            break;
        }
    }
    if (k == num_in) {
        q[0] = p[t];
        return 1;
    }
    q[1] = q[k];
    int m = 2;
    for (int i = k + 1; i < num_in; i++) {
        while (m > 1 && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
            m--;
        }
        q[m++] = q[i];
    }

    return m;
}

static float polygon_area(const Point(&q)[24], const int& m) {
    if (m <= 2) {
        return 0;
    }

    float area = 0;
    for (int i = 1; i < m - 1; i++) {
        area += fabs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
    }

    return area / 2.0;
}

static float rotated_boxes_intersection(const Object& box1, const Object& box2) {
    Point intersectPts[24], orderedPts[24];

    Point pts1[4];
    Point pts2[4];
    get_rotated_vertices(box1, pts1);
    get_rotated_vertices(box2, pts2);

    int num = get_intersection_points(pts1, pts2, intersectPts);

    if (num <= 2) {
        return 0.0;
    }

    int num_convex = convex_hull_graham(intersectPts, num, orderedPts);
    return polygon_area(orderedPts, num_convex);
}

static void nms_sorted_bboxes(const std::vector<Object> & objects, std::vector<int> & picked, float nms_threshold)
{
	picked.clear();

	const int n = objects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
        float width = objects[i].xmax - objects[i].xmin;
        float height = objects[i].ymax - objects[i].ymin;
        areas[i] = width * height;
	}

	for (int i = 0; i < n; i++)
	{
		const Object& a = objects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Object& b = objects[picked[j]];

			// intersection over union
			//float inter_area = intersection_area(a, b);
			float inter_area = rotated_boxes_intersection(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			float IoU = inter_area / union_area;
			if (IoU > nms_threshold)
            {
				keep = 0;
                break;
            }
		}

		if (keep)
			picked.push_back(i);
	}
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

#if defined(RV1106_1103)
static int process_i8_rv1106(int8_t *box_tensor, int32_t box_zp, float box_scale,
                             int8_t *score_tensor, int32_t score_zp, float score_scale,
                             int8_t *angle_tensor, int32_t angle_zp, float angle_scale,
                             int grid_h, int grid_w, int stride, int dfl_len,
                             std::vector<Object> &boxes, float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;
            int8_t max_score = -score_zp;

            offset = offset * OBJ_CLASS_NUM;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                if ((score_tensor[offset + c] > score_thres_i8) && (score_tensor[offset + c] > max_score)){
                    max_score = score_tensor[offset + c];
                    max_class_id = c;
                    // float temp = deqnt_affine_to_f32(max_score, score_zp, score_scale);
                    // printf("max_class_id=%d; max_score=%f\n", max_class_id, temp);
                }
            }

            // compute box
            if (max_score > score_thres_i8) {
                int angle_offset = i * grid_w + j;
                offset = angle_offset * 4 * dfl_len;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset + k], box_zp, box_scale);
                }

                compute_dfl(before_dfl, dfl_len, box);

                int8_t ang = angle_tensor[angle_offset];

                float x1, y1, x2, y2;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;

                Object obj;
                obj.xmin = x1;
                obj.ymin = y1;
                obj.xmax = x2;
                obj.ymax = y2;
                obj.label = max_class_id;
                obj.angle = deqnt_affine_to_f32(ang, angle_zp, angle_scale);
                obj.prob  = deqnt_affine_to_f32(max_score, score_zp, score_scale);
                boxes.push_back(obj);

                validCount ++;
            }
        }
    }
    // printf("validCount=%d\n", validCount);
    // printf("grid h-%d, w-%d, stride %d\n", grid_h, grid_w, stride);
    return validCount;
}
#endif

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
#if defined(RV1106_1103)
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)outputs;
#else
    rknn_output *_outputs = (rknn_output *)outputs;
#endif
    std::vector<Object> filterBoxes;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int num_c = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    int output_per_branch = app_ctx->io_num.n_output / 3;

    for (int i = 0; i < 3; i++)
    {
#if defined(RV1106_1103) 
        int dfl_len = app_ctx->output_attrs[0].dims[3] / 4;
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;
        int angle_idx = i * output_per_branch + 2;
        grid_h = app_ctx->output_attrs[box_idx].dims[1];
        grid_w = app_ctx->output_attrs[box_idx].dims[2];
        stride = model_in_h / grid_h;
        
        //RV1106 only support i8
        if (app_ctx->is_quant) {
            validCount += process_i8_rv1106(
                                    (int8_t *)_outputs[box_idx]->virt_addr, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                    (int8_t *)_outputs[score_idx]->virt_addr, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                    (int8_t *)_outputs[angle_idx]->virt_addr, app_ctx->output_attrs[angle_idx].zp, app_ctx->output_attrs[angle_idx].scale,
                                    grid_h, grid_w, stride, dfl_len, filterBoxes, conf_threshold);
        }
#else     
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        //  if (app_ctx->is_quant)
        // {
        //     validCount += process_i8((int8_t *)_outputs[i].buf, (int *)anchor[i], grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
        //                              classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale);
        // }
        // else
        // {
        //     validCount += process_fp32((float *)_outputs[i].buf, (int *)anchor[i], grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
        //                                classId, conf_threshold);
        // }
#endif
    }
    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }

    qsort_descent_inplace(filterBoxes);

    // apply nms with nms_threshold
    std::vector<int> picked;
    // 根据八个点进行 NMS
    nms_sorted_bboxes(filterBoxes, picked, nms_threshold);

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    int count = picked.size();
    for (int i = 0; i < count; ++i)
    {
        if (last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }

        float x1 = filterBoxes[picked[i]].xmin - letter_box->x_pad;
        float y1 = filterBoxes[picked[i]].ymin - letter_box->y_pad;
        float x2 = filterBoxes[picked[i]].xmax - letter_box->x_pad;
        float y2 = filterBoxes[picked[i]].ymax - letter_box->y_pad;
        int id = filterBoxes[picked[i]].label;
        float obj_conf = filterBoxes[picked[i]].prob;
        float angle = filterBoxes[picked[i]].angle;

        Object orignobj;
        orignobj.xmin = (float)(x1 / letter_box->scale);
        orignobj.ymin = (float)(y1 / letter_box->scale);
        orignobj.xmax = (float)(x2 / letter_box->scale);
        orignobj.ymax = (float)(y2 / letter_box->scale);
        orignobj.angle = angle;

        get_rotated_vertices(orignobj, od_results->results[last_count].ptsl);

        od_results->results[last_count].prob= obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

const char *coco_cls_to_name(int cls_id)
{

    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
