#include "yolo.h"

int main(int argc, char const *argv[]) {

  Yolo yolo("./cfg/yolo-refine.conv_adas.json",
            "./models/yolo-refine_10000.weights", 0.2);

  yolo.Setup();
  //yolo.Test("./data/demo_5.png");
  // yolo.BatchTest("./data/testlist.txt", false);
   //yolo.VideoTest("./data/set07V000.avi", true);
   yolo.Demo(0, false);
  yolo.Release();
  return 0;
}
