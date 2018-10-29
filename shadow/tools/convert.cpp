#include "transformer.hpp"

using namespace Shadow;

int main(int argc, char const* argv[]) {
  std::string save_path("models/mtcnn");
  std::string model("models/mtcnn/mtcnn_merged.shadowmodel");

  shadow::MetaNetParam meta_net_param;
  IO::ReadProtoFromBinaryFile(model, &meta_net_param);
  WriteProtoToFiles(meta_net_param.network(0), save_path, "mtcnn_merged_p");
  WriteProtoToFiles(meta_net_param.network(1), save_path, "mtcnn_merged_r");
  WriteProtoToFiles(meta_net_param.network(2), save_path, "mtcnn_merged_o");

  return 0;
}
