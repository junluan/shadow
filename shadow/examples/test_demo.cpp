#include "demo_classification.hpp"
#include "demo_detection.hpp"

int main(int argc, char const *argv[]) {
  std::string model("models/ssd/shadow/adas_model_finetune_reduce_3.shadowmodel");
  std::string test_image("data/static/demo_6.png");
  std::string test_list("data/static/demo_list.txt");

  Shadow::DemoDetection demo("ssd");
  demo.Setup({model}, {1});
  demo.Test(test_image);
  //demo.BatchTest(test_list, false);
  //demo.CameraTest(0);

  return 0;
}
