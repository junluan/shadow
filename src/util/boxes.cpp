#include "boxes.hpp"

float BoxArea(const Box &box) { return box.w * box.h; }

float BoxesOverlap(float x1, float w1, float x2, float w2) {
  float left = x1 > x2 ? x1 : x2;
  float right = x1 + w1 < x2 + w2 ? x1 + w1 : x2 + w2;
  return right - left;
}

float Boxes::BoxesIntersection(const Box &box_a, const Box &box_b) {
  float width = BoxesOverlap(box_a.x, box_a.w, box_b.x, box_b.w);
  float height = BoxesOverlap(box_a.y, box_a.h, box_b.y, box_b.h);
  if (width < 0 || height < 0)
    return 0;
  return width * height;
}

float Boxes::BoxesUnion(const Box &box_a, const Box &box_b) {
  return BoxArea(box_a) + BoxArea(box_b) - BoxesIntersection(box_a, box_b);
}

void MergeBoxes(const Box &old_box, Box *new_box, float smooth) {
  new_box->x = old_box.x + (new_box->x - old_box.x) * smooth;
  new_box->y = old_box.y + (new_box->y - old_box.y) * smooth;
  new_box->w = old_box.w + (new_box->w - old_box.w) * smooth;
  new_box->h = old_box.h + (new_box->h - old_box.h) * smooth;
}

float Boxes::BoxesIoU(const Box &box_a, const Box &box_b) {
  return BoxesIntersection(box_a, box_b) / BoxesUnion(box_a, box_b);
}

VecBox Boxes::BoxesNMS(const std::vector<VecBox> &Bboxes, float iou_threshold) {
  VecBox boxes;
  for (int i = 0; i < Bboxes.size(); ++i) {
    for (int j = 0; j < Bboxes[i].size(); ++j) {
      if (Bboxes[i][j].class_index != -1)
        boxes.push_back(Bboxes[i][j]);
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].class_index == -1)
      continue;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (boxes[j].class_index == -1 ||
          boxes[i].class_index != boxes[j].class_index)
        continue;
      if (BoxesIoU(boxes[i], boxes[j]) > iou_threshold) {
        float smooth = boxes[i].score / (boxes[i].score + boxes[j].score);
        MergeBoxes(boxes[j], &boxes[i], smooth);
        boxes[j].class_index = -1;
        continue;
      }
      float in = BoxesIntersection(boxes[i], boxes[j]);
      float cover_i = in / BoxArea(boxes[i]);
      float cover_j = in / BoxArea(boxes[j]);
      if (cover_i > cover_j && cover_i > 0.7)
        boxes[i].class_index = -1;
      if (cover_i < cover_j && cover_j > 0.7)
        boxes[j].class_index = -1;
    }
  }
  VecBox out_boxes;
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].class_index != -1)
      out_boxes.push_back(boxes[i]);
  }
  return out_boxes;
}

void Boxes::SmoothBoxes(const VecBox &old_boxes, VecBox *new_boxes,
                        float smooth) {
  for (int i = 0; i < new_boxes->size(); ++i) {
    Box &newBox = (*new_boxes)[i];
    for (int j = 0; j < old_boxes.size(); ++j) {
      Box oldBox = old_boxes[j];
      if (BoxesIoU(oldBox, newBox) > 0.7) {
        MergeBoxes(oldBox, &newBox, smooth);
        break;
      }
    }
  }
}

void Boxes::AmendBoxes(std::vector<VecBox> *boxes, int height, int width,
                       VecRectF crops) {
  for (int i = 0; i < crops.size(); ++i) {
    for (int b = 0; b < (*boxes)[i].size(); ++b) {
      (*boxes)[i][b].x += crops[i].x * width;
      (*boxes)[i][b].y += crops[i].y * height;
    }
  }
}

#ifdef USE_OpenCV
void Boxes::SelectRoI(int event, int x, int y, int flags, void *roi) {
  Box *roi_ = reinterpret_cast<Box *>(roi);
  switch (event) {
  case cv::EVENT_LBUTTONDOWN: {
    std::cout << "Mouse Pressed" << std::endl;
    roi_->x = x;
    roi_->y = y;
  }
  case cv::EVENT_LBUTTONUP: {
    std::cout << "Mouse LBUTTON Released" << std::endl;
    roi_->w = x - roi_->x;
    roi_->h = y - roi_->y;
  }
  default:
    return;
  }
}
#endif
