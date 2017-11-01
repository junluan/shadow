#include "caffe2shadow.hpp"

using namespace Shadow;

int main(int argc, char const* argv[]) {
  std::string model_root("models/ssd");
  std::string save_path(model_root + "/shadow");

  // mtcnn info
  NetInfo net_mtcnn_r, net_mtcnn_p, net_mtcnn_o;
  MetaNetInfo meta_net_mtcnn_info;

  net_mtcnn_r.num_class = {0};
  net_mtcnn_r.input_shape = {{1, 3, 360, 360}};
  net_mtcnn_r.mean_value = {127.5f};
  net_mtcnn_r.scale = 0.0078125f;
  net_mtcnn_r.out_blob = {"conv4-2", "prob1"};

  net_mtcnn_p.num_class = {0};
  net_mtcnn_p.input_shape = {{50, 3, 24, 24}};
  net_mtcnn_p.mean_value = {127.5f};
  net_mtcnn_p.scale = 0.0078125f;
  net_mtcnn_p.out_blob = {"conv5-2", "prob1"};

  net_mtcnn_o.num_class = {0};
  net_mtcnn_o.input_shape = {{20, 3, 48, 48}};
  net_mtcnn_o.mean_value = {127.5f};
  net_mtcnn_o.scale = 0.0078125f;
  net_mtcnn_o.out_blob = {"conv6-2", "conv6-3", "prob1"};

  meta_net_mtcnn_info.version = "0.0.1";
  meta_net_mtcnn_info.method = "mtcnn";
  meta_net_mtcnn_info.model_name = {"det1", "det2", "det3"};
  meta_net_mtcnn_info.save_name = "mtcnn";
  meta_net_mtcnn_info.network = {net_mtcnn_r, net_mtcnn_p, net_mtcnn_o};

  // ssd info
  NetInfo net_ssd;
  MetaNetInfo meta_net_ssd_info;

  net_ssd.num_class = {3};
  net_ssd.input_shape = {{1, 3, 300, 300}};
  net_ssd.mean_value = {103.94f, 116.78f, 123.68f};
  net_ssd.out_blob = {"mbox_loc", "mbox_conf_flatten", "mbox_priorbox"};

  meta_net_ssd_info.version = "0.0.1";
  meta_net_ssd_info.method = "ssd";
  meta_net_ssd_info.model_name = {"adas_model_finetune_reduce_3"};
  meta_net_ssd_info.save_name = "adas_model_finetune_reduce_3";
  meta_net_ssd_info.network = {net_ssd};

  // faster rcnn info
  NetInfo net_faster;
  MetaNetInfo meta_net_faster_info;

  net_faster.num_class = {21};
  net_faster.input_shape = {{1, 3, 224, 224}, {1, 3}};
  net_faster.mean_value = {102.9801f, 115.9465f, 122.7717f};
  net_faster.out_blob = {"bbox_pred", "cls_prob"};

  meta_net_faster_info.version = "0.0.1";
  meta_net_faster_info.method = "faster";
  meta_net_faster_info.model_name = {"VGG16_faster_rcnn_final"};
  meta_net_faster_info.save_name = "VGG16_faster_rcnn_final";
  meta_net_faster_info.network = {net_faster};

  const auto& meta_net_info = meta_net_ssd_info;

  std::vector<caffe::NetParameter> caffe_deploys, caffe_models;
  for (const auto& model_name : meta_net_info.model_name) {
    std::string deploy_file(model_root + "/" + model_name + ".prototxt");
    std::string model_file(model_root + "/" + model_name + ".caffemodel");
    caffe::NetParameter caffe_deploy, caffe_model;
    IO::ReadProtoFromTextFile(deploy_file, &caffe_deploy);
    IO::ReadProtoFromBinaryFile(model_file, &caffe_model);
    caffe_deploys.push_back(caffe_deploy);
    caffe_models.push_back(caffe_model);
  }

  shadow::MetaNetParam shadow_net;
  Caffe2Shadow::ConvertCaffe(caffe_deploys, caffe_models, meta_net_info,
                             &shadow_net);

  WriteProtoToBinary(shadow_net, save_path, meta_net_info.save_name);

  return 0;
}
