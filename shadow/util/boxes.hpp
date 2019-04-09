#ifndef SHADOW_UTIL_BOXES_HPP
#define SHADOW_UTIL_BOXES_HPP

#include "type.hpp"

namespace Shadow {

template <class Dtype>
class Box {
 public:
  Box() = default;
  Box(Dtype xmin_t, Dtype ymin_t, Dtype xmax_t, Dtype ymax_t)
      : xmin(xmin_t), ymin(ymin_t), xmax(xmax_t), ymax(ymax_t) {}

  RectI RectInt() const { return RectI(xmin, ymin, xmax - xmin, ymax - ymin); }
  RectF RectFloat() const {
    return RectF(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  Dtype xmin = 0, ymin = 0, xmax = 0, ymax = 0;
  float score = 0;
  int label = -1;
};

using BoxI = Box<int>;
using BoxF = Box<float>;

using VecBoxI = std::vector<BoxI>;
using VecBoxF = std::vector<BoxF>;

namespace Boxes {

template <typename Dtype>
void Clip(const Box<Dtype> &box, Box<Dtype> *clip_box, Dtype min, Dtype max);

template <typename Dtype>
Dtype Size(const Box<Dtype> &box);

template <typename Dtype>
float Intersection(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
float Union(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
float IoU(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<Box<Dtype>> &boxes,
                            float iou_threshold);
template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<std::vector<Box<Dtype>>> &Gboxes,
                            float iou_threshold);

template <typename Dtype>
void Smooth(const Box<Dtype> &old_box, Box<Dtype> *new_box, float smooth);
template <typename Dtype>
void Smooth(const std::vector<Box<Dtype>> &old_boxes,
            std::vector<Box<Dtype>> *new_boxes, float smooth);

template <typename Dtype>
void Amend(std::vector<std::vector<Box<Dtype>>> *Gboxes, const VecRectF &crops,
           int height = 1, int width = 1);

}  // namespace Boxes

}  // namespace Shadow

#endif  // SHADOW_UTIL_BOXES_HPP
