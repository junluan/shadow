#ifndef SHADOW_CORE_NETWORK_HPP
#define SHADOW_CORE_NETWORK_HPP

#include "params.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Shadow {

class Network {
 public:
  Network();

  void Setup(int device_id = 0);

  void LoadModel(const shadow::NetParam &net_param);
  void LoadModel(const void *proto_data, int proto_size);
  void LoadModel(const std::string &proto_bin);
  void LoadModel(const std::string &proto_str,
                 const std::vector<const void *> &weights);
  void LoadModel(const std::string &proto_str, const void *weights_data);

  void Forward(const std::map<std::string, float *> &data_map,
               const std::map<std::string, std::vector<int>> &shape_map = {});

  template <typename T>
  const T *GetBlobDataByName(const std::string &blob_name);
  template <typename T>
  std::vector<int> GetBlobShapeByName(const std::string &blob_name) const;

  const std::vector<std::string> &in_blob() const;
  const std::vector<std::string> &out_blob() const;

  bool has_argument(const std::string &name) const;
  template <typename T>
  T get_single_argument(const std::string &name, const T &default_value) const;
  template <typename T>
  std::vector<T> get_repeated_argument(
      const std::string &name, const std::vector<T> &default_value = {}) const;

 private:
  class NetworkImpl;
  std::shared_ptr<NetworkImpl> engine_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_HPP
