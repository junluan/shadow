#ifndef SHADOW_UTIL_BOXES_HPP
#define SHADOW_UTIL_BOXES_HPP

#include "shadow/util/util.hpp"

template <class Dtype>
class Box {
 public:
  Box() {}
  Box(Dtype x_t, Dtype y_t, Dtype w_t, Dtype h_t)
      : x(x_t), y(y_t), w(w_t), h(h_t) {}
  Box(const Box<int> &box) : x(box.x), y(box.y), w(box.w), h(box.h) {}
  Box(const Box<float> &box) : x(box.x), y(box.y), w(box.w), h(box.h) {}

  RectI RectInt() { return RectI(x, y, w, h); }
  RectF RectFloat() { return RectF(x, y, w, h); }

  float score;
  int class_index;
  Dtype x, y, w, h;
};

typedef Box<int> BoxI;
typedef Box<float> BoxF;

typedef std::vector<BoxI> VecBoxI;
typedef std::vector<BoxF> VecBoxF;

namespace Boxes {

template <typename Dtype>
float Intersection(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
float Union(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
float IoU(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<std::vector<Box<Dtype>>> &Bboxes,
                            float iou_threshold);

template <typename Dtype>
void Smooth(const std::vector<Box<Dtype>> &old_boxes,
            std::vector<Box<Dtype>> *new_boxes, float smooth);

template <typename Dtype>
void Amend(std::vector<std::vector<Box<Dtype>>> *boxes, const VecRectF &crops,
           int height = 0, int width = 0);

}  // namespace Boxes

#endif  // SHADOW_UTIL_BOXES_HPP
