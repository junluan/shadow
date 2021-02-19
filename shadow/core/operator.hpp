#ifndef SHADOW_CORE_OPERATOR_HPP_
#define SHADOW_CORE_OPERATOR_HPP_

#include "blas.hpp"
#include "blob.hpp"
#include "external.hpp"
#include "helper.hpp"
#include "params.hpp"
#include "registry.hpp"
#include "workspace.hpp"

#include "util/json_helper.hpp"
#include "util/log.hpp"
#include "util/type.hpp"
#include "util/util.hpp"

namespace Shadow {

class Operator {
 public:
  Operator(const shadow::OpParam& op_param, Workspace* ws);
  virtual ~Operator() = default;

  virtual void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
                   std::vector<std::shared_ptr<Blob>>& outputs) = 0;

  bool has_argument(const std::string& name) const {
    return arg_helper_.HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string& name, const T& default_value) const {
    return arg_helper_.GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  std::vector<T> get_repeated_argument(
      const std::string& name, const std::vector<T>& default_value = {}) const {
    return arg_helper_.GetRepeatedArgument<T>(name, default_value);
  }
  template <typename T>
  std::vector<T> get_repeated_argument(const std::string& name,
                                       const T& default_value) const {
    const auto& val = arg_helper_.GetRepeatedArgument<T>(name);
    if (val.empty()) {
      return std::vector<T>(
          1, arg_helper_.GetSingleArgument<T>(name, default_value));
    } else {
      return val;
    }
  }

  Workspace* ws() const { return ws_; }

  const shadow::OpParam& op_param() const { return op_param_; }

  const std::string& name() const { return op_param_.name(); }
  const std::string& type() const { return op_param_.type(); }

  std::string debug_log(
      const std::vector<std::shared_ptr<Blob>>& inputs,
      const std::vector<std::shared_ptr<Blob>>& outputs) const;

 protected:
  Workspace* ws_ = nullptr;

 private:
  DEFINE_JSON_SERIALIZATION(op_param_);

  shadow::OpParam op_param_;
  ArgumentHelper arg_helper_;
};

std::shared_ptr<Operator> CreateOperator(const shadow::OpParam& op_param,
                                         Workspace* ws);

SHADOW_DECLARE_REGISTRY(OperatorRegistry, Operator, const shadow::OpParam&,
                        Workspace*);

#define REGISTER_OPERATOR(name, ...) \
  SHADOW_REGISTER_CLASS(OperatorRegistry, name, __VA_ARGS__)

}  // namespace Shadow

#endif  // SHADOW_CORE_OPERATOR_HPP_
