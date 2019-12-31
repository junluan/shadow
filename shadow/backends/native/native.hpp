#ifndef SHADOW_BACKENDS_NATIVE_NATIVE_HPP
#define SHADOW_BACKENDS_NATIVE_NATIVE_HPP

#include "core/backend.hpp"
#include "core/operator.hpp"

namespace Shadow {

class Native : public Backend {
 public:
  Native(const ArgumentHelper &arguments, Workspace *ws) : Backend(ws) {
    device_input_ = arguments.GetSingleArgument<bool>("device_input", false);
  }

  void LoadModel(const shadow::NetParam &net_param) override;

  void LoadModel(const void *proto_data, int proto_size);
  void LoadModel(const std::string &proto_bin);
  void LoadModel(const std::string &proto_str,
                 const std::vector<const void *> &weights);
  void LoadModel(const std::string &proto_str, const void *weights_data);

  void Forward(
      const std::map<std::string, float *> &data_map,
      const std::map<std::string, std::vector<int>> &shape_map) override;

  void SaveEngine(const std::string &save_path,
                  std::vector<char> *save_data) override;

 private:
  static void LoadProtoData(const void *proto_data, int proto_size,
                            shadow::NetParam *net_param);
  static void LoadProtoBin(const std::string &proto_bin,
                           shadow::NetParam *net_param);
  static void LoadProtoStrOrText(const std::string &proto_str_or_text,
                                 shadow::NetParam *net_param);

  void Initial();

  void CopyWeights(const std::vector<const void *> &weights);
  void CopyWeights(const void *weights_data);

  bool device_input_ = false;

  shadow::NetParam net_param_;
  std::vector<std::shared_ptr<Operator>> ops_;
};

}  // namespace Shadow

#endif  // SHADOW_BACKENDS_NATIVE_NATIVE_HPP
