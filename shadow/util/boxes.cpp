#include "boxes.hpp"
#include "log.hpp"

namespace Shadow {

namespace Boxes {

template <typename T>
void Clip(const Box<T>& box, Box<T>* clip_box, T min, T max) {
  clip_box->xmin = std::max(std::min(box.xmin, max), min);
  clip_box->ymin = std::max(std::min(box.ymin, max), min);
  clip_box->xmax = std::max(std::min(box.xmax, max), min);
  clip_box->ymax = std::max(std::min(box.ymax, max), min);
}

template <typename T>
T Size(const Box<T>& box) {
  return (box.xmax - box.xmin) * (box.ymax - box.ymin);
}

template <typename T>
inline T BorderOverlap(T a1, T a2, T b1, T b2) {
  auto left = a1 > b1 ? a1 : b1, right = a2 < b2 ? a2 : b2;
  return right - left;
}

template <typename T>
float Intersection(const Box<T>& box_a, const Box<T>& box_b) {
  auto width = BorderOverlap(box_a.xmin, box_a.xmax, box_b.xmin, box_b.xmax);
  auto height = BorderOverlap(box_a.ymin, box_a.ymax, box_b.ymin, box_b.ymax);
  if (width < 0 || height < 0) return 0;
  return width * height;
}

template <typename T>
float Union(const Box<T>& box_a, const Box<T>& box_b) {
  return Size(box_a) + Size(box_b) - Intersection(box_a, box_b);
}

template <typename T>
float IoU(const Box<T>& box_a, const Box<T>& box_b) {
  return Intersection(box_a, box_b) / Union(box_a, box_b);
}

template <typename T>
inline bool SortBoxesAscend(const Box<T>& box_a, const Box<T>& box_b) {
  return box_a.score < box_b.score;
}

template <typename T>
inline bool SortBoxesDescend(const Box<T>& box_a, const Box<T>& box_b) {
  return box_a.score > box_b.score;
}

template <typename T>
std::vector<Box<T>> NMS(const std::vector<Box<T>>& boxes, float iou_threshold) {
  auto all_boxes = boxes;
  std::stable_sort(all_boxes.begin(), all_boxes.end(), SortBoxesDescend<T>);
  for (int i = 0; i < all_boxes.size(); ++i) {
    auto& box_i = all_boxes[i];
    if (box_i.label == -1) continue;
    for (int j = i + 1; j < all_boxes.size(); ++j) {
      auto& box_j = all_boxes[j];
      if (box_j.label == -1 || box_i.label != box_j.label) continue;
      if (IoU(box_i, box_j) > iou_threshold) {
        box_j.label = -1;
      }
    }
  }
  std::vector<Box<T>> out_boxes;
  for (const auto& box : all_boxes) {
    if (box.label != -1) {
      out_boxes.push_back(box);
    }
  }
  all_boxes.clear();
  return out_boxes;
}

template <typename T>
std::vector<Box<T>> NMS(const std::vector<std::vector<Box<T>>>& Gboxes,
                        float iou_threshold) {
  std::vector<Box<T>> all_boxes;
  for (const auto& boxes : Gboxes) {
    for (const auto& box : boxes) {
      if (box.label != -1) {
        all_boxes.push_back(box);
      }
    }
  }
  return NMS<T>(all_boxes, iou_threshold);
}

template <typename T>
void Smooth(const Box<T>& old_box, Box<T>* new_box, float smooth) {
  new_box->xmin = old_box.xmin + (new_box->xmin - old_box.xmin) * smooth;
  new_box->ymin = old_box.ymin + (new_box->ymin - old_box.ymin) * smooth;
  new_box->xmax = old_box.xmax + (new_box->xmax - old_box.xmax) * smooth;
  new_box->ymax = old_box.ymax + (new_box->ymax - old_box.ymax) * smooth;
}

template <typename T>
void Smooth(const std::vector<Box<T>>& old_boxes,
            std::vector<Box<T>>* new_boxes, float smooth) {
  for (auto& new_box : *new_boxes) {
    for (auto& old_box : old_boxes) {
      if (IoU(old_box, new_box) > 0.7) {
        Smooth(old_box, &new_box, smooth);
        break;
      }
    }
  }
}

template <typename T>
void Amend(std::vector<std::vector<Box<T>>>* Gboxes, const VecRectF& crops,
           int height, int width) {
  CHECK_EQ(Gboxes->size(), crops.size());
  for (int i = 0; i < crops.size(); ++i) {
    auto& boxes = Gboxes->at(i);
    const auto& crop = crops[i];
    bool normalize = crop.h <= 1 || crop.w <= 1;
    if (normalize) {
      CHECK_GT(height, 1);
      CHECK_GT(width, 1);
    }
    float x_off = normalize ? crop.x * width : crop.x;
    float y_off = normalize ? crop.y * height : crop.y;
    for (auto& box : boxes) {
      box.xmin += x_off;
      box.xmax += x_off;
      box.ymin += y_off;
      box.ymax += y_off;
    }
  }
}

// Explicit instantiation
template void Clip(const BoxI&, BoxI*, int, int);
template void Clip(const BoxF&, BoxF*, float, float);

template int Size(const BoxI&);
template float Size(const BoxF&);

template float Intersection(const BoxI&, const BoxI&);
template float Intersection(const BoxF&, const BoxF&);

template float Union(const BoxI&, const BoxI&);
template float Union(const BoxF&, const BoxF&);

template float IoU(const BoxI&, const BoxI&);
template float IoU(const BoxF&, const BoxF&);

template VecBoxI NMS(const VecBoxI&, float);
template VecBoxF NMS(const VecBoxF&, float);

template VecBoxI NMS(const std::vector<VecBoxI>&, float);
template VecBoxF NMS(const std::vector<VecBoxF>&, float);

template void Smooth(const BoxI&, BoxI*, float);
template void Smooth(const BoxF&, BoxF*, float);

template void Smooth(const VecBoxI&, VecBoxI*, float);
template void Smooth(const VecBoxF&, VecBoxF*, float);

template void Amend(std::vector<VecBoxI>*, const VecRectF&, int, int);
template void Amend(std::vector<VecBoxF>*, const VecRectF&, int, int);

}  // namespace Boxes

}  // namespace Shadow
