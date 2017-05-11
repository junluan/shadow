#ifndef SHADOW_UTIL_BOXES_HPP
#define SHADOW_UTIL_BOXES_HPP

#include "type.hpp"

namespace Shadow {

template <class Dtype>
class Box {
 public:
  Box() {}
  Box(Dtype xmin_t, Dtype ymin_t, Dtype xmax_t, Dtype ymax_t)
      : xmin(xmin_t), ymin(ymin_t), xmax(xmax_t), ymax(ymax_t) {}
  Box(const Box<int> &box) { *this = box; }
  Box(const Box<float> &box) { *this = box; }

  Box &operator=(const Box<int> &box) {
    xmin = static_cast<Dtype>(box.xmin);
    ymin = static_cast<Dtype>(box.ymin);
    xmax = static_cast<Dtype>(box.xmax);
    ymax = static_cast<Dtype>(box.ymax);
    score = box.score;
    label = box.label;
    return *this;
  }
  Box &operator=(const Box<float> &box) {
    xmin = static_cast<Dtype>(box.xmin);
    ymin = static_cast<Dtype>(box.ymin);
    xmax = static_cast<Dtype>(box.xmax);
    ymax = static_cast<Dtype>(box.ymax);
    score = box.score;
    label = box.label;
    return *this;
  }

  RectI RectInt() const { return RectI(xmin, ymin, xmax - xmin, ymax - ymin); }
  RectF RectFloat() const {
    return RectF(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  float score;
  int label;
  Dtype xmin, ymin, xmax, ymax;
};

typedef Box<int> BoxI;
typedef Box<float> BoxF;

typedef std::vector<BoxI> VecBoxI;
typedef std::vector<BoxF> VecBoxF;

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
std::vector<Box<Dtype>> NMS(const std::vector<std::vector<Box<Dtype>>> &Bboxes,
                            float iou_threshold);

template <typename Dtype>
void Smooth(const Box<Dtype> &old_boxes, Box<Dtype> *new_boxes, float smooth);
template <typename Dtype>
void Smooth(const std::vector<Box<Dtype>> &old_boxes,
            std::vector<Box<Dtype>> *new_boxes, float smooth);

template <typename Dtype>
void Amend(std::vector<std::vector<Box<Dtype>>> *Bboxes, const VecRectF &crops,
           int height = 1, int width = 1);

}  // namespace Boxes

}  // namespace Shadow

#endif  // SHADOW_UTIL_BOXES_HPP
