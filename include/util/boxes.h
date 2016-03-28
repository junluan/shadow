#ifndef SHADOW_BOXES_H
#define SHADOW_BOXES_H

#include <vector>

struct Box {
  float x, y, w, h;
  std::vector<float> probability;
  int classindex = -1;
};

typedef std::vector<Box> VecBox;

class Boxes {
public:
  Boxes();
  ~Boxes();

  static void BoxesNMS(VecBox &boxes, int num_classes, float iou_threshold);
  static float BoxesIoU(Box boxA, Box boxB);
  static Box BoxFromFloat(float *value);
};

#endif // SHADOW_BOXES_H
