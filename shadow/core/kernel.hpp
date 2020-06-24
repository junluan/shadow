#ifndef SHADOW_CORE_KERNEL_HPP_
#define SHADOW_CORE_KERNEL_HPP_

#include "blob.hpp"
#include "external.hpp"
#include "registry.hpp"
#include "workspace.hpp"

#include "util/type.hpp"

namespace Shadow {

class Kernel {
 public:
  virtual ~Kernel() = default;

  virtual DeviceType device_type() const = 0;
  virtual std::string kernel_type() const = 0;
};

VecString GetKernelKeys(DeviceType device_type);

VecString GetKernelKeys(const std::string& op_type, DeviceType device_type);

std::shared_ptr<Kernel> CreateKernel(const std::string& op_type,
                                     DeviceType device_type,
                                     const std::string& kernel_type = "");

SHADOW_DECLARE_REGISTRY(KernelRegistry, Kernel);

#define REGISTER_OP_KERNEL(name, ...) \
  SHADOW_REGISTER_CLASS(KernelRegistry, name, __VA_ARGS__)

#define REGISTER_OP_KERNEL_DEFAULT(name, ...) \
  REGISTER_OP_KERNEL(name(Default), __VA_ARGS__)

#define REGISTER_OP_KERNEL_DNNL(name, ...) \
  REGISTER_OP_KERNEL(name(DNNL), __VA_ARGS__)

#define REGISTER_OP_KERNEL_CUDNN(name, ...) \
  REGISTER_OP_KERNEL(name(CUDNN), __VA_ARGS__)

}  // namespace Shadow

#endif  // SHADOW_CORE_KERNEL_HPP_
