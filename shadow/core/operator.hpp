#ifndef SHADOW_CORE_OPERATOR_HPP_
#define SHADOW_CORE_OPERATOR_HPP_

#include "blas.hpp"
#include "blob.hpp"
#include "common.hpp"
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
  virtual ~Operator();

  virtual void Forward() = 0;

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

  const std::string& name() const { return op_name_; }
  const std::string& type() const { return op_type_; }

  std::shared_ptr<Blob> bottoms(int n) const {
    auto blob = ws_->GetBlob(bottoms_name(n));
    CHECK_NOTNULL(blob);
    return blob;
  }
  const std::string& bottoms_name(int n) const {
    CHECK(check_index(n, bottoms_size()));
    return bottom_names_[n];
  }
  int bottoms_size() const { return static_cast<int>(bottom_names_.size()); }

  std::shared_ptr<Blob> tops(int n) const {
    auto blob = ws_->GetBlob(tops_name(n));
    CHECK_NOTNULL(blob);
    return blob;
  }
  const std::string& tops_name(int n) const {
    CHECK(check_index(n, tops_size()));
    return top_names_[n];
  }
  int tops_size() const { return static_cast<int>(top_names_.size()); }

  std::string debug_log() const;

 protected:
  Workspace* ws_ = nullptr;

 private:
  bool check_index(int i, int size) const { return i >= 0 && i < size; }

  DEFINE_JSON_SERIALIZATION(op_param_);

  shadow::OpParam op_param_;
  ArgumentHelper arg_helper_;

  std::string op_name_, op_type_;
  VecString bottom_names_, top_names_;

  DISABLE_COPY_AND_ASSIGN(Operator);
};

Operator* CreateOperator(const shadow::OpParam& op_param, Workspace* ws);

SHADOW_DECLARE_REGISTRY(OperatorRegistry, Operator, const shadow::OpParam&,
                        Workspace*);

#define REGISTER_OPERATOR(name, ...) \
  SHADOW_REGISTER_CLASS(OperatorRegistry, name, __VA_ARGS__)

}  // namespace Shadow

#endif  // SHADOW_CORE_OPERATOR_HPP_
