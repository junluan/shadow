#include "caffe2shadow.hpp"

using namespace Shadow::Caffe2Shadow;
using namespace Shadow::IO;

int main(int argc, char const* argv[]) {
  std::string model_root = "models/ssd";
  std::string model_name = "adas_model_finetune_reduce";

  std::string save_path = model_root + "/adas";

  std::string deploy_file = model_root + "/" + model_name + ".prototxt";
  std::string model_file = model_root + "/" + model_name + ".caffemodel";

  caffe::NetParameter caffe_deploy, caffe_model;
  ReadProtoFromTextFile(deploy_file, &caffe_deploy);
  ReadProtoFromBinaryFile(model_file, &caffe_model);

  const std::vector<int> input_shape{1, 3, 300, 300};

  shadow::NetParam shadow_net;
  Convert(caffe_deploy, caffe_model, input_shape, &shadow_net);

  WriteProtoToBinary(shadow_net, save_path, model_name);

  //WriteProtoToFiles(shadow_net, save_path, model_name);

  return 0;
}
