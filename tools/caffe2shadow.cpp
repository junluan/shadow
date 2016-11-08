#include "caffe2shadow.hpp"

using namespace Caffe2Shadow;

int main(int argc, char const* argv[]) {
  std::string deploy_file = "models/ssd/adas_deploy.prototxt";
  std::string model_file = "models/ssd/adas_model.caffemodel";

  std::string root = "models/ssd/adas";
  std::string model_name = "adas";

  caffe::NetParameter caffe_deploy, caffe_model;
  shadow::NetParameter shadow_net;

  shadow_net.mutable_input_shape()->add_dim(1);
  shadow_net.mutable_input_shape()->add_dim(3);
  shadow_net.mutable_input_shape()->add_dim(300);
  shadow_net.mutable_input_shape()->add_dim(300);

  IO::ReadProtoFromTextFile(deploy_file, &caffe_deploy);
  IO::ReadProtoFromBinaryFile(model_file, &caffe_model);

  Convert(caffe_deploy, caffe_model, &shadow_net);

  WriteProtoToBinary(shadow_net, root, model_name + "_model");

  //WriteProtoToFiles(shadow_net, root, model_name);

  return 0;
}
