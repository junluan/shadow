#ifndef SHADOW_UTIL_BOXES_HPP
#define SHADOW_UTIL_BOXES_HPP

#include "shadow/util/util.hpp"

#include <vector>

#if defined(USE_OpenCV)
#include <opencv2/opencv.hpp>
#endif

class Box : public Rect<float> {
public:
  Box() {}
  Box(float x_t, float y_t, float w_t, float h_t)
      : Rect<float>(x_t, y_t, w_t, h_t) {}
  float score;
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

#if defined(USE_OpenCV)
  static void SelectRoI(int event, int x, int y, int flags, void *roi);
#endif
};

#endif // SHADOW_UTIL_BOXES_HPP
