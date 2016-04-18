#include "boxes.h"

float BoxArea(Box box) { return box.w * box.h; }

float BoxesOverlap(float x1, float w1, float x2, float w2) {
  float left = x1 > x2 ? x1 : x2;
  float right = x1 + w1 < x2 + w2 ? x1 + w1 : x2 + w2;
  return right - left;
}

float BoxesIntersection(Box box_a, Box box_b) {
  float width = BoxesOverlap(box_a.x, box_a.w, box_b.x, box_b.w);
  float height = BoxesOverlap(box_a.y, box_a.h, box_b.y, box_b.h);
  if (width < 0 || height < 0)
    return 0;
  return width * height;
}

float BoxesUnion(Box box_a, Box box_b) {
  return BoxArea(box_a) + BoxArea(box_b) - BoxesIntersection(box_a, box_b);
}

float Boxes::BoxesIoU(Box box_a, Box box_b) {
  return BoxesIntersection(box_a, box_b) / BoxesUnion(box_a, box_b);
}

void Boxes::BoxesNMS(VecBox &boxes, float iou_threshold) {
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i].class_index == -1)
      continue;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (boxes[j].class_index == -1 ||
          boxes[i].class_index != boxes[j].class_index)
        continue;
      if (BoxesIoU(boxes[i], boxes[j]) > iou_threshold) {
        if (boxes[i].score < boxes[j].score)
          boxes[i].class_index = -1;
        else
          boxes[j].class_index = -1;
        continue;
      }
      float in = BoxesIntersection(boxes[i], boxes[j]);
      float cover_i = in / BoxArea(boxes[i]);
      float cover_j = in / BoxArea(boxes[j]);
      if (cover_i > cover_j && cover_i > 0.7) {
        boxes[i].class_index = -1;
      }
      if (cover_i < cover_j && cover_j > 0.7) {
        boxes[j].class_index = -1;
      }
    }
  }
}

void MergeBoxes(Box &oldBox, Box &newBox, float smooth) {
  newBox.x = oldBox.x + (newBox.x - oldBox.x) * smooth;
  newBox.y = oldBox.y + (newBox.y - oldBox.y) * smooth;
  newBox.w = oldBox.w + (newBox.w - oldBox.w) * smooth;
  newBox.h = oldBox.h + (newBox.h - oldBox.h) * smooth;
}

void Boxes::SmoothBoxes(VecBox &oldBoxes, VecBox &newBoxes, float smooth) {
  for (int i = 0; i < newBoxes.size(); ++i) {
    Box &newBox = newBoxes[i];
    for (int j = 0; j < oldBoxes.size(); ++j) {
      Box oldBox = oldBoxes[j];
      if (BoxesIoU(newBox, oldBox) > 0.7) {
        MergeBoxes(oldBox, newBox, smooth);
        break;
      }
    }
  }
}

void Boxes::AmendBoxes(VecBox &boxes, Box *roi) {
  if (roi == nullptr)
    return;
  for (int i = 0; i < boxes.size(); ++i) {
    boxes[i].x += roi->x;
    boxes[i].y += roi->y;
  }
}

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
