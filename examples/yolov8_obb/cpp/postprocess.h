#ifndef _RKNN_YOLOV8_OBB_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_OBB_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 4
#define NMS_THRESH 0.10
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)
	
struct Object {
	float xmin;
	float xmax;
	float ymin;
	float ymax;
	float prob;
	float angle;
	int label;
};

struct Point {
	float x, y;
	Point(const float& px = 0, const float& py = 0) : x(px), y(py) {}
	Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
	Point& operator+=(const Point& p) {
		x += p.x;
		y += p.y;
		return *this;
	}
	Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
	Point operator*(const float coeff) const { return Point(x * coeff, y * coeff); }
};

struct RotatedBox {
    Point center;
    float width, height;
    float angle;  // in degrees
};

typedef struct {
    Point ptsl[4];
    float prob;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int init_post_process();
void deinit_post_process();
const char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
Point intersection(const Point& a, const Point& b, const Point& c, const Point& d);
void deinitPostProcess();
#endif //_RKNN_YOLOV8_OBB_DEMO_POSTPROCESS_H_
