#include "boxes.h"

Boxes::Boxes() {}
Boxes::~Boxes() {}

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

void Boxes::BoxesNMS(VecBox &boxes, int num_classes, float iou_threshold) {
  for (int i = 0; i < boxes.size(); ++i) {
    bool is_object = false;
    for (int k = 0; k < num_classes; ++k)
      is_object |= (boxes[i].probability[k] > 0);
    if (!is_object)
      continue;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (BoxesIoU(boxes[i], boxes[j]) > iou_threshold) {
        for (int k = 0; k < num_classes; ++k) {
          if (boxes[i].probability[k] < boxes[j].probability[k])
            boxes[i].probability[k] = 0;
          else
            boxes[j].probability[k] = 0;
        }
      }
    }
    if (boxes[i].w / boxes[i].h < 0.3) {
      // for (int k = 0; k < num_classes; ++k)
      //  boxes[i].probability[k] = 0;
    }

    int index = 0;
    float value = boxes[i].probability[0];
    for (int k = 1; k < num_classes; ++k) {
      if (boxes[i].probability[k] > value) {
        index = k;
        value = boxes[i].probability[k];
      }
    }
    if (value > 0)
      boxes[i].classindex = index;
  }
}

Box Boxes::BoxFromFloat(float *value) {
  Box box;
  box.x = value[0];
  box.y = value[1];
  box.w = value[2];
  box.h = value[3];
  return box;
}
