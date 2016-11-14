#include "shadow/util/boxes.hpp"

namespace Boxes {

template <typename Dtype>
inline Dtype BoxArea(const Box<Dtype> &box) {
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
  return BoxArea(box_a) + BoxArea(box_b) - Intersection(box_a, box_b);
}

template <typename Dtype>
float IoU(const Box<Dtype> &box_a, const Box<Dtype> &box_b) {
  return Intersection(box_a, box_b) / Union(box_a, box_b);
}

template <typename Dtype>
inline void MergeBoxes(const Box<Dtype> &old_box, Box<Dtype> *new_box,
                       float smooth) {
  new_box->xmin = old_box.xmin + (new_box->xmin - old_box.xmin) * smooth;
  new_box->ymin = old_box.ymin + (new_box->ymin - old_box.ymin) * smooth;
  new_box->xmax = old_box.xmax + (new_box->xmax - old_box.xmax) * smooth;
  new_box->ymax = old_box.ymax + (new_box->ymax - old_box.ymax) * smooth;
}

template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<std::vector<Box<Dtype>>> &Bboxes,
                            float iou_threshold) {
  std::vector<Box<Dtype>> boxes;
  for (int i = 0; i < Bboxes.size(); ++i) {
    for (int j = 0; j < Bboxes[i].size(); ++j) {
      if (Bboxes[i][j].label != -1) boxes.push_back(Bboxes[i][j]);
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].label == -1) continue;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (boxes[j].label == -1 || boxes[i].label != boxes[j].label) continue;
      if (IoU(boxes[i], boxes[j]) > iou_threshold) {
        float smooth = boxes[i].score / (boxes[i].score + boxes[j].score);
        MergeBoxes(boxes[j], &boxes[i], smooth);
        boxes[j].label = -1;
        continue;
      }
      float in = Intersection(boxes[i], boxes[j]);
      float cover_i = in / BoxArea(boxes[i]);
      float cover_j = in / BoxArea(boxes[j]);
      if (cover_i > cover_j && cover_i > 0.7) boxes[i].label = -1;
      if (cover_i < cover_j && cover_j > 0.7) boxes[j].label = -1;
    }
  }
  std::vector<Box<Dtype>> out_boxes;
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].label != -1) out_boxes.push_back(boxes[i]);
  }
  boxes.clear();
  return out_boxes;
}

template <typename Dtype>
void Smooth(const std::vector<Box<Dtype>> &old_boxes,
            std::vector<Box<Dtype>> *new_boxes, float smooth) {
  for (int i = 0; i < new_boxes->size(); ++i) {
    Box<Dtype> &newBox = new_boxes->at(i);
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
void Amend(std::vector<std::vector<Box<Dtype>>> *Bboxes, const VecRectF &crops,
           int height, int width) {
  CHECK_EQ(Bboxes->size(), crops.size());
  for (int i = 0; i < crops.size(); ++i) {
    std::vector<Box<Dtype>> &boxes = Bboxes->at(i);
    const RectF &crop = crops[i];
    float x_off = crop.x <= 1 ? crop.x * width : crop.x;
    float y_off = crop.y <= 1 ? crop.y * height : crop.y;
    for (int b = 0; b < boxes.size(); ++b) {
      Box<Dtype> &box = boxes.at(b);
      box.xmin += x_off;
      box.xmax += x_off;
      box.ymin += y_off;
      box.ymax += y_off;
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

template void Amend(std::vector<VecBoxI> *Bboxes, const VecRectF &crops,
                    int height, int width);
template void Amend(std::vector<VecBoxF> *Bboxes, const VecRectF &crops,
                    int height, int width);

}  // namespace Boxes
