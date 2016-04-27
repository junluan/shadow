#include "yolo.hpp"

int main(int argc, char const *argv[]) {

  Yolo yolo("./cfg/yolo-refine.conv_adas.json",
            "./models/yolo-refine.conv_adas.weights", 0.2);

  Box roi;
  roi.x = 700;
  roi.y = 500;
  roi.w = 660;
  roi.h = 480;
  yolo.Setup();
  yolo.Test("./data/demo_6.png");
  //yolo.BatchTest("./data/testlist.txt", true);
  //yolo.VideoTest("./data/traffic/set07V000.avi", true, true);
  //yolo.Demo(0, true);
  yolo.Release();
  return 0;
}
