#include "shadow/util/boxes.hpp"

namespace Boxes {

template <typename Dtype>
inline Dtype BoxArea(const Box<Dtype> &box) {
  return box.w * box.h;
}

template <typename Dtype>
inline Dtype BoxesOverlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype left = x1 > x2 ? x1 : x2;
  Dtype right = x1 + w1 < x2 + w2 ? x1 + w1 : x2 + w2;
  return right - left;
}

template <typename Dtype>
float Intersection(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  Dtype width = BoxesOverlap(box_a.x, box_a.w, box_b.x, box_b.w);
  Dtype height = BoxesOverlap(box_a.y, box_a.h, box_b.y, box_b.h);
  if (width < 0 || height < 0) return 0;
  return width * height;
}

template <typename Dtype>
float Union(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  return BoxArea(box_a) + BoxArea(box_b) - Intersection(box_a, box_b);
}

template <typename Dtype>
float IoU(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  return Intersection(box_a, box_b) / Union(box_a, box_b);
}

template <typename Dtype>
inline void MergeBoxes(const Box<Dtype> &old_box, Box<Dtype> *new_box,
                       float smooth) {
  new_box->x = old_box.x + (new_box->x - old_box.x) * smooth;
  new_box->y = old_box.y + (new_box->y - old_box.y) * smooth;
  new_box->w = old_box.w + (new_box->w - old_box.w) * smooth;
  new_box->h = old_box.h + (new_box->h - old_box.h) * smooth;
}

template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<std::vector<Box<Dtype>>> &Bboxes,
                            float iou_threshold) {
  std::vector<Box<Dtype>> boxes;
  for (int i = 0; i < Bboxes.size(); ++i) {
    for (int j = 0; j < Bboxes[i].size(); ++j) {
      if (Bboxes[i][j].class_index != -1) boxes.push_back(Bboxes[i][j]);
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].class_index == -1) continue;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (boxes[j].class_index == -1 ||
          boxes[i].class_index != boxes[j].class_index)
        continue;
      if (IoU(boxes[i], boxes[j]) > iou_threshold) {
        float smooth = boxes[i].score / (boxes[i].score + boxes[j].score);
        MergeBoxes(boxes[j], &boxes[i], smooth);
        boxes[j].class_index = -1;
        continue;
      }
      float in = Intersection(boxes[i], boxes[j]);
      float cover_i = in / BoxArea(boxes[i]);
      float cover_j = in / BoxArea(boxes[j]);
      if (cover_i > cover_j && cover_i > 0.7) boxes[i].class_index = -1;
      if (cover_i < cover_j && cover_j > 0.7) boxes[j].class_index = -1;
    }
  }
  std::vector<Box<Dtype>> out_boxes;
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].class_index != -1) out_boxes.push_back(boxes[i]);
  }
  boxes.clear();
  return out_boxes;
}

template <typename Dtype>
void Smooth(const std::vector<Box<Dtype>> &old_boxes,
            std::vector<Box<Dtype>> *new_boxes, float smooth) {
  for (int i = 0; i < new_boxes->size(); ++i) {
    Box<Dtype> &newBox = (*new_boxes)[i];
    for (int j = 0; j < old_boxes.size(); ++j) {
      const Box<Dtype> &oldBox = old_boxes[j];
      if (IoU(oldBox, newBox) > 0.7) {
        MergeBoxes(oldBox, &newBox, smooth);
        break;
      }
    }
  }
}

template <typename Dtype>
void Amend(std::vector<std::vector<Box<Dtype>>> *boxes, const VecRectF &crops,
           int height, int width) {
  for (int i = 0; i < crops.size(); ++i) {
    for (int b = 0; b < (*boxes)[i].size(); ++b) {
      (*boxes)[i][b].x += crops[i].x <= 1 ? crops[i].x * width : crops[i].x;
      (*boxes)[i][b].y += crops[i].y <= 1 ? crops[i].y * height : crops[i].y;
    }
  }
}

// Explicit instantiation
template float Intersection(const BoxI &box_a, const BoxI &box_b);
template float Intersection(const BoxF &box_a, const BoxF &box_b);

template float Union(const BoxI &box_a, const BoxI &box_b);
template float Union(const BoxF &box_a, const BoxF &box_b);

template float IoU(const BoxI &box_a, const BoxI &box_b);
template float IoU(const BoxF &box_a, const BoxF &box_b);

template void Smooth(const VecBoxI &old_boxes, VecBoxI *new_boxes,
                     float smooth);
template void Smooth(const VecBoxF &old_boxes, VecBoxF *new_boxes,
                     float smooth);

template VecBoxI NMS(const std::vector<VecBoxI> &Bboxes, float iou_threshold);
template VecBoxF NMS(const std::vector<VecBoxF> &Bboxes, float iou_threshold);

template void Amend(std::vector<VecBoxI> *boxes, const VecRectF &crops,
                    int height, int width);
template void Amend(std::vector<VecBoxF> *boxes, const VecRectF &crops,
                    int height, int width);

}  // namespace Boxes
