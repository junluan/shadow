#include "network.hpp"
#include "network_impl.hpp"

namespace Shadow {

Network::Network() {
  if (engine_ == nullptr) {
    engine_ = std::make_shared<NetworkImpl>();
  }
}

void Network::Setup(int device_id) { engine_->Setup(device_id); }

void Network::LoadModel(const shadow::NetParam &net_param) {
  engine_->LoadModel(net_param);
}

void Network::LoadModel(const void *proto_data, int proto_size) {
  engine_->LoadModel(proto_data, proto_size);
}

void Network::LoadModel(const std::string &proto_bin) {
  engine_->LoadModel(proto_bin);
}

void Network::LoadModel(const std::string &proto_str,
                        const std::vector<const void *> &weights) {
  engine_->LoadModel(proto_str, weights);
}

void Network::LoadModel(const std::string &proto_str,
                        const void *weights_data) {
  engine_->LoadModel(proto_str, weights_data);
}

void Network::LoadXModel(const shadow::NetParam &net_param,
                         const ArgumentHelper &arguments) {
  engine_->LoadXModel(net_param, arguments);
}

void Network::Forward(
    const std::map<std::string, float *> &data_map,
    const std::map<std::string, std::vector<int>> &shape_map) {
  engine_->Forward(data_map, shape_map);
}

#define INSTANTIATE_GET_BLOB(T)                                          \
  template <>                                                            \
  const T *Network::GetBlobDataByName<T>(const std::string &blob_name) { \
    return engine_->GetBlobDataByName<T>(blob_name);                     \
  }                                                                      \
  template <>                                                            \
  std::vector<int> Network::GetBlobShapeByName<T>(                       \
      const std::string &blob_name) const {                              \
    return engine_->GetBlobShapeByName<T>(blob_name);                    \
  }

INSTANTIATE_GET_BLOB(float);
INSTANTIATE_GET_BLOB(int);
#undef INSTANTIATE_GET_BLOB

const std::vector<std::string> &Network::in_blob() const {
  return engine_->in_blob();
}

const std::vector<std::string> &Network::out_blob() const {
  return engine_->out_blob();
}

bool Network::has_argument(const std::string &name) const {
  return engine_->has_argument(name);
}

#define INSTANTIATE_ARGUMENT(T)                                             \
  template <>                                                               \
  T Network::get_single_argument<T>(const std::string &name,                \
                                    const T &default_value) const {         \
    return engine_->get_single_argument<T>(name, default_value);            \
  }                                                                         \
  template <>                                                               \
  std::vector<T> Network::get_repeated_argument<T>(                         \
      const std::string &name, const std::vector<T> &default_value) const { \
    return engine_->get_repeated_argument<T>(name, default_value);          \
  }

INSTANTIATE_ARGUMENT(float);
INSTANTIATE_ARGUMENT(int);
INSTANTIATE_ARGUMENT(bool);
INSTANTIATE_ARGUMENT(std::string);
#undef INSTANTIATE_ARGUMENT

}  // namespace Shadow
