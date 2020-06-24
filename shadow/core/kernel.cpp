#include "kernel.hpp"

namespace Shadow {

const std::map<DeviceType, std::string> device_type_map{
    {DeviceType::kCPU, "CPU"}, {DeviceType::kGPU, "GPU"}};

const std::map<DeviceType, VecString> kernel_type_map{
    {DeviceType::kCPU, {"DNNL", "NNPACK", "Default"}},
    {DeviceType::kGPU, {"CUDNN", "Default"}}};

VecString get_kernel_keys(const std::string& hint) {
  static const auto& all_kernel_keys = KernelRegistry()->Keys();
  VecString kernel_keys;
  for (const auto& kernel_key : all_kernel_keys) {
    if (kernel_key.find(hint) != std::string::npos) {
      kernel_keys.push_back(kernel_key);
    }
  }
  return kernel_keys;
}

VecString GetKernelKeys(DeviceType device_type) {
  return get_kernel_keys(device_type_map.at(device_type));
}

VecString GetKernelKeys(const std::string& op_type, DeviceType device_type) {
  return get_kernel_keys(op_type + device_type_map.at(device_type));
}

std::string get_best_kernel_type(const std::string& op_type,
                                 DeviceType device_type) {
  const auto& kernel_keys = GetKernelKeys(op_type, device_type);
  for (const auto& kernel_type : kernel_type_map.at(device_type)) {
    for (const auto& kernel_key : kernel_keys) {
      if (kernel_key.find(kernel_type) != std::string::npos) {
        return kernel_type;
      }
    }
  }
  return "None";
}

std::shared_ptr<Kernel> CreateKernel(const std::string& op_type,
                                     DeviceType device_type,
                                     const std::string& kernel_type) {
  auto best_kernel_type = kernel_type.empty()
                              ? get_best_kernel_type(op_type, device_type)
                              : kernel_type;
  const auto& kernel_key =
      op_type + device_type_map.at(device_type) + "(" + best_kernel_type + ")";
  auto kernel = std::shared_ptr<Kernel>(KernelRegistry()->Create(kernel_key));
  LOG_IF(FATAL, kernel == nullptr)
      << "Kernel type: " << kernel_key << " is not registered";
  return kernel;
}

SHADOW_DEFINE_REGISTRY(KernelRegistry, Kernel);

}  // namespace Shadow
