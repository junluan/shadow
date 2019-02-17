#ifndef SHADOW_CORE_OPERATOR_HPP
#define SHADOW_CORE_OPERATOR_HPP

#include "blas.hpp"
#include "blob.hpp"
#include "helper.hpp"
#include "params.hpp"
#include "registry.hpp"
#include "workspace.hpp"

#include "util/json_helper.hpp"
#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

class Operator {
 public:
  explicit Operator(const shadow::OpParam &op_param, Workspace *ws);
  virtual ~Operator();

  virtual void Forward() = 0;

  bool has_argument(const std::string &name) const {
    return arg_helper_.HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string &name, const T &default_value) const {
    return arg_helper_.template GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  const std::vector<T> get_repeated_argument(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    return arg_helper_.template GetRepeatedArgument<T>(name, default_value);
  }

  const std::string &name() const { return op_name_; }
  const std::string &type() const { return op_type_; }

  template <typename T>
  const Blob<T> *bottoms(int n) const {
    CHECK(check_index(n, bottoms_size()));
    return op_ws_->GetBlob<T>(bottom_names_[n]);
  }
  template <typename T>
  Blob<T> *mutable_bottoms(int n) {
    return const_cast<Blob<T> *>(
        static_cast<const Operator *>(this)->bottoms<T>(n));
  }
  template <typename T>
  void add_bottoms(const std::string &bottom_name) {
    op_ws_->CreateBlob<T>(bottom_name);
    bottom_names_.push_back(bottom_name);
  }
  template <typename T>
  void set_bottoms(int n, int count, const T *data) {
    CHECK(check_index(n, bottoms_size()));
    mutable_bottoms<T>(n)->set_data(data, count);
  }
  const std::string &bottoms_name(int n) const {
    CHECK(check_index(n, bottoms_size()));
    return bottom_names_[n];
  }
  const std::string bottoms_type(int n) const {
    CHECK(check_index(n, bottoms_size()));
    return op_ws_->GetBlobType(bottom_names_[n]);
  }
  int bottoms_size() const { return static_cast<int>(bottom_names_.size()); }

  template <typename T>
  const Blob<T> *tops(int n) const {
    CHECK(check_index(n, tops_size()));
    return op_ws_->GetBlob<T>(top_names_[n]);
  }
  template <typename T>
  Blob<T> *mutable_tops(int n) {
    return const_cast<Blob<T> *>(
        static_cast<const Operator *>(this)->tops<T>(n));
  }
  template <typename T>
  void add_tops(const std::string &top_name) {
    op_ws_->CreateBlob<T>(top_name);
    top_names_.push_back(top_name);
  }
  template <typename T>
  void set_tops(int n, int count, const T *data) {
    CHECK(check_index(n, tops_size()));
    mutable_tops<T>(n)->set_data(data, count);
  }
  const std::string tops_name(int n) const {
    CHECK(check_index(n, tops_size()));
    return top_names_[n];
  }
  const std::string tops_type(int n) const {
    CHECK(check_index(n, tops_size()));
    return op_ws_->GetBlobType(top_names_[n]);
  }
  int tops_size() const { return static_cast<int>(top_names_.size()); }

  const std::string debug_log() const;

 protected:
  std::string op_name_, op_type_;
  Workspace *op_ws_ = nullptr;

 private:
  bool check_index(int i, int size) const { return i >= 0 && i < size; }

  DEFINE_JSON_SERIALIZATION(op_param_);

  shadow::OpParam op_param_;
  ArgumentHelper arg_helper_;

  VecString bottom_names_, top_names_;

  DISABLE_COPY_AND_ASSIGN(Operator);
};

Operator *CreateOperator(const shadow::OpParam &op_param, Workspace *ws);

SHADOW_DECLARE_REGISTRY(OperatorRegistry, Operator, const shadow::OpParam &,
                        Workspace *);
#define REGISTER_OPERATOR(name, ...) \
  SHADOW_REGISTER_CLASS(OperatorRegistry, name, __VA_ARGS__)

class StaticLinkingProtector {
 public:
  StaticLinkingProtector() {
    const auto &registered_ops = OperatorRegistry()->Keys();
    LOG_IF(FATAL, registered_ops.empty())
        << "You might have made a build error: the Shadow library does not "
           "seem to be linked with whole-static library option. To do so, use "
           "-Wl,-force_load (clang) or -Wl,--whole-archive (gcc) to link the "
           "Shadow library.";
  }
};

}  // namespace Shadow

#endif  // SHADOW_CORE_OPERATOR_HPP
