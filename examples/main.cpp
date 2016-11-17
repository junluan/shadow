#include "demo.hpp"

int main(int argc, char const *argv[]) {
  std::string model = "models/ssd/adas/adas_finetune_model.shadowmodel";
  std::string test_image = "data/demo_6.png";
  std::string test_list = "data/demo_list.txt";

  Demo demo;
  demo.Setup(model, 1);
  demo.Test(test_image);
  //demo.BatchTest(test_list, true);
  //demo.CameraTest(0);
  demo.Release();

  return 0;
}
