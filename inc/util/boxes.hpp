#ifndef SHADOW_BOXES_HPP
#define SHADOW_BOXES_HPP

#include "util.hpp"

#include <vector>

#ifdef USE_OpenCV
#include <opencv2/opencv.hpp>
#endif

class Box {
public:
  Box() {}
  Box(float x_t, float y_t, float w_t, float h_t) {
    x = x_t;
    y = y_t;
    w = w_t;
    h = h_t;
  }
  float x, y, w, h, score;
  int class_index;
};

typedef std::vector<Box> VecBox;

class Boxes {
public:
  static float BoxesIntersection(const Box &boxA, const Box &boxB);
  static float BoxesUnion(const Box &box_a, const Box &box_b);
  static float BoxesIoU(const Box &boxA, const Box &boxB);
  static VecBox BoxesNMS(const std::vector<VecBox> &Bboxes,
                         float iou_threshold);
  static void SmoothBoxes(const VecBox &old_boxes, VecBox *new_boxes,
                          float smooth);
  static void AmendBoxes(std::vector<VecBox> *boxes, int height, int width,
                         VecRectF crops);

#ifdef USE_OpenCV
  static void SelectRoI(int event, int x, int y, int flags, void *roi);
#endif
};

#endif // SHADOW_BOXES_HPP
