#include "transformer.hpp"

using namespace Shadow;

int main(int argc, char const* argv[]) {
  std::string model("models/ssd/adas_model_finetune_reduce_3_merged.shadowmodel");
  std::string save_path("models/ssd");

  shadow::NetParam net_param;
  IO::ReadProtoFromBinaryFile(model, &net_param);
  WriteProtoToFiles(net_param, save_path, "adas_model_finetune_reduce_3_merged");

  return 0;
}
