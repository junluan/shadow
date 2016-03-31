#include "yolo.h"

int main(int argc, char const *argv[]) {

  Yolo yolo("./cfg/yolo-refine.conv_adas.json",
            "./models/yolo-refine_final.weights", 0.2);

  yolo.Setup();
  //yolo.Test("./data/demo_4.png");
  // yolo.BatchTest("./data/testlist.txt", false);
  // yolo.VideoTest("./data/traffic/2016-03-07-08-50-03.avi", true);
   yolo.Demo(0, false);
  yolo.Release();
  return 0;
}
