#include "boxes.hpp"
#include "log.hpp"

namespace Shadow {

namespace Boxes {

template <typename Dtype>
void Clip(const Box<Dtype> &box, Box<Dtype> *clip_box, Dtype min, Dtype max) {
  clip_box->xmin = std::max(std::min(box.xmin, max), min);
  clip_box->ymin = std::max(std::min(box.ymin, max), min);
  clip_box->xmax = std::max(std::min(box.xmax, max), min);
  clip_box->ymax = std::max(std::min(box.ymax, max), min);
}

template <typename Dtype>
Dtype Size(const Box<Dtype> &box) {
  return (box.xmax - box.xmin) * (box.ymax - box.ymin);
}

template <typename Dtype>
inline Dtype BorderOverlap(Dtype a1, Dtype a2, Dtype b1, Dtype b2) {
  Dtype left = a1 > b1 ? a1 : b1;
  Dtype right = a2 < b2 ? a2 : b2;
  return right - left;
}

template <typename Dtype>
float Intersection(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  Dtype width = BorderOverlap(box_a.xmin, box_a.xmax, box_b.xmin, box_b.xmax);
  Dtype height = BorderOverlap(box_a.ymin, box_a.ymax, box_b.ymin, box_b.ymax);
  if (width < 0 || height < 0) return 0;
  return width * height;
}

template <typename Dtype>
float Union(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  return Size(box_a) + Size(box_b) - Intersection(box_a, box_b);
}

template <typename Dtype>
float IoU(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  return Intersection(box_a, box_b) / Union(box_a, box_b);
}

template <typename Dtype>
inline bool SortBoxesAscend(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  return box_a.score < box_b.score;
}

template <typename Dtype>
inline bool SortBoxesDescend(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  return box_a.score > box_b.score;
}

template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<std::vector<Box<Dtype>>> &Bboxes,
                            float iou_threshold) {
  std::vector<Box<Dtype>> all_boxes;
  for (const auto &boxes : Bboxes) {
    for (const auto &box : boxes) {
      if (box.label != -1) {
        all_boxes.push_back(box);
      }
    }
  }
  std::stable_sort(all_boxes.begin(), all_boxes.end(), SortBoxesDescend<Dtype>);
  for (int i = 0; i < all_boxes.size(); ++i) {
    Box<Dtype> &box_i = all_boxes[i];
    if (box_i.label == -1) continue;
    for (int j = i + 1; j < all_boxes.size(); ++j) {
      Box<Dtype> &box_j = all_boxes[j];
      if (box_j.label == -1 || box_i.label != box_j.label) continue;
      if (IoU(box_i, box_j) > iou_threshold) {
        float smooth = box_i.score / (box_i.score + box_j.score);
        Smooth(box_j, &box_i, smooth);
        box_j.label = -1;
        continue;
      }
      float in = Intersection(box_i, box_j);
      float cover_i = in / Size(box_i), cover_j = in / Size(box_j);
      if (cover_i > cover_j && cover_i > 0.7) box_i.label = -1;
      if (cover_i < cover_j && cover_j > 0.7) box_j.label = -1;
    }
  }
  std::vector<Box<Dtype>> out_boxes;
  for (const auto &box : all_boxes) {
    if (box.label != -1) {
      out_boxes.push_back(box);
    }
  }
  all_boxes.clear();
  return out_boxes;
}

template <typename Dtype>
void Smooth(const Box<Dtype> &old_box, Box<Dtype> *new_box, float smooth) {
  new_box->xmin = old_box.xmin + (new_box->xmin - old_box.xmin) * smooth;
  new_box->ymin = old_box.ymin + (new_box->ymin - old_box.ymin) * smooth;
  new_box->xmax = old_box.xmax + (new_box->xmax - old_box.xmax) * smooth;
  new_box->ymax = old_box.ymax + (new_box->ymax - old_box.ymax) * smooth;
}

template <typename Dtype>
void Smooth(const std::vector<Box<Dtype>> &old_boxes,
            std::vector<Box<Dtype>> *new_boxes, float smooth) {
  for (auto &new_box : *new_boxes) {
    for (auto &old_box : old_boxes) {
      if (IoU(old_box, new_box) > 0.7) {
        Smooth(old_box, &new_box, smooth);
        break;
      }
    }
  }
}

template <typename Dtype>
void Amend(std::vector<std::vector<Box<Dtype>>> *Bboxes, const VecRectF &crops,
           int height, int width) {
  CHECK_EQ(Bboxes->size(), crops.size());
  for (int i = 0; i < crops.size(); ++i) {
    std::vector<Box<Dtype>> &boxes = Bboxes->at(i);
    const RectF &crop = crops[i];
    bool normalize = crop.h <= 1 || crop.w <= 1;
    if (normalize) {
      CHECK_GT(height, 1);
      CHECK_GT(width, 1);
    }
    float x_off = normalize ? crop.x * width : crop.x;
    float y_off = normalize ? crop.y * height : crop.y;
    for (auto &box : boxes) {
      box.xmin += x_off;
      box.xmax += x_off;
      box.ymin += y_off;
      box.ymax += y_off;
    }
  }
}

// Explicit instantiation
template void Clip(const BoxI &box, BoxI *clip_box, int min, int max);
template void Clip(const BoxF &box, BoxF *clip_box, float min, float max);

template int Size(const BoxI &box);
template float Size(const BoxF &box);

template float Intersection(const BoxI &box_a, const BoxI &box_b);
template float Intersection(const BoxF &box_a, const BoxF &box_b);

template float Union(const BoxI &box_a, const BoxI &box_b);
template float Union(const BoxF &box_a, const BoxF &box_b);

template float IoU(const BoxI &box_a, const BoxI &box_b);
template float IoU(const BoxF &box_a, const BoxF &box_b);

template VecBoxI NMS(const std::vector<VecBoxI> &Bboxes, float iou_threshold);
template VecBoxF NMS(const std::vector<VecBoxF> &Bboxes, float iou_threshold);

template void Smooth(const BoxI &old_boxes, BoxI *new_boxes, float smooth);
template void Smooth(const BoxF &old_boxes, BoxF *new_boxes, float smooth);

template void Smooth(const VecBoxI &old_boxes, VecBoxI *new_boxes,
                     float smooth);
template void Smooth(const VecBoxF &old_boxes, VecBoxF *new_boxes,
                     float smooth);

template void Amend(std::vector<VecBoxI> *Bboxes, const VecRectF &crops,
                    int height, int width);
template void Amend(std::vector<VecBoxF> *Bboxes, const VecRectF &crops,
                    int height, int width);

}  // namespace Boxes

}  // namespace Shadow
