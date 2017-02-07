#include "caffe2shadow.hpp"

using namespace Caffe2Shadow;

int main(int argc, char const* argv[]) {
  std::string model_root = "models/ssd";
  std::string model_name = "adas_model_finetune_reduce";

  std::string save_prefix = "adas";
  std::string save_path = model_root + "/" + save_prefix;

  std::string deploy_file = model_root + "/" + model_name + ".prototxt";
  std::string model_file = model_root + "/" + model_name + ".caffemodel";

  caffe::NetParameter caffe_deploy, caffe_model;
  IO::ReadProtoFromTextFile(deploy_file, &caffe_deploy);
  IO::ReadProtoFromBinaryFile(model_file, &caffe_model);

  shadow::NetParameter shadow_net;

  const auto input_shape = std::vector<int>{1, 3, 300, 300};

  Convert(caffe_deploy, caffe_model, input_shape, &shadow_net);

  WriteProtoToBinary(shadow_net, save_path, model_name);

  //WriteProtoToFiles(shadow_net, save_path, save_prefix);

  return 0;
}
