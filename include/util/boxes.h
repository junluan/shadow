#ifndef SHADOW_BOXES_H
#define SHADOW_BOXES_H

#include <vector>

struct Box {
  float x, y, w, h, score;
  int class_index;
};

typedef std::vector<Box> VecBox;

class Boxes {
public:
  static float BoxesIoU(Box boxA, Box boxB);
  static void BoxesNMS(VecBox &boxes, float iou_threshold);
  static void SmoothBoxes(VecBox &oldBoxes, VecBox &newBoxes, float smooth);
  static Box BoxFromFloat(float *value);
};

#endif // SHADOW_BOXES_H
