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
  ArgumentHelper arguments;
  arguments.AddSingleArgument<std::string>("backend_type", "Native");

  engine_->LoadXModel(net_param, arguments);
}

void Network::LoadXModel(const shadow::NetParam &net_param,
                         const ArgumentHelper &arguments) {
  engine_->LoadXModel(net_param, arguments);
}

void Network::Forward(
    const std::map<std::string, void *> &data_map,
    const std::map<std::string, std::vector<int>> &shape_map) {
  engine_->Forward(data_map, shape_map);
}

#define INSTANTIATE_GET_BLOB(T)                                        \
  template <>                                                          \
  const T *Network::GetBlobDataByName<T>(const std::string &blob_name, \
                                         const std::string &locate) {  \
    return engine_->GetBlobDataByName<T>(blob_name, locate);           \
  }                                                                    \
  template <>                                                          \
  std::vector<int> Network::GetBlobShapeByName<T>(                     \
      const std::string &blob_name) const {                            \
    return engine_->GetBlobShapeByName<T>(blob_name);                  \
  }

INSTANTIATE_GET_BLOB(int);
INSTANTIATE_GET_BLOB(float);
INSTANTIATE_GET_BLOB(unsigned char);
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
