#ifndef SHADOW_UTIL_BOXES_HPP
#define SHADOW_UTIL_BOXES_HPP

#include "type.hpp"

namespace Shadow {

template <typename T>
class Box {
 public:
  Box() = default;
  Box(T xmin_t, T ymin_t, T xmax_t, T ymax_t)
      : xmin(xmin_t), ymin(ymin_t), xmax(xmax_t), ymax(ymax_t) {}

  RectI RectInt() const { return RectI(xmin, ymin, xmax - xmin, ymax - ymin); }
  RectF RectFloat() const {
    return RectF(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  T xmin = 0, ymin = 0, xmax = 0, ymax = 0;
  float score = 0;
  int label = -1;
};

using BoxI = Box<int>;
using BoxF = Box<float>;

using VecBoxI = std::vector<BoxI>;
using VecBoxF = std::vector<BoxF>;

namespace Boxes {

template <typename T>
void Clip(const Box<T> &box, Box<T> *clip_box, T min, T max);

template <typename T>
T Size(const Box<T> &box);

template <typename T>
float Intersection(const Box<T> &box_a, const Box<T> &box_b);

template <typename T>
float Union(const Box<T> &box_a, const Box<T> &box_b);

template <typename T>
float IoU(const Box<T> &box_a, const Box<T> &box_b);

template <typename T>
std::vector<Box<T>> NMS(const std::vector<Box<T>> &boxes, float iou_threshold);
template <typename T>
std::vector<Box<T>> NMS(const std::vector<std::vector<Box<T>>> &Gboxes,
                        float iou_threshold);

template <typename T>
void Smooth(const Box<T> &old_box, Box<T> *new_box, float smooth);
template <typename T>
void Smooth(const std::vector<Box<T>> &old_boxes,
            std::vector<Box<T>> *new_boxes, float smooth);

template <typename T>
void Amend(std::vector<std::vector<Box<T>>> *Gboxes, const VecRectF &crops,
           int height = 1, int width = 1);

}  // namespace Boxes

}  // namespace Shadow

#endif  // SHADOW_UTIL_BOXES_HPP
