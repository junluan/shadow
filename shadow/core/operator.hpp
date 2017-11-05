#ifndef SHADOW_CORE_OPERATOR_HPP
#define SHADOW_CORE_OPERATOR_HPP

#include "blas.hpp"
#include "blob.hpp"
#include "params.hpp"
#include "registry.hpp"
#include "workspace.hpp"

#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

class Operator {
 public:
  explicit Operator(const shadow::OpParam &op_param, Workspace *ws);
  virtual ~Operator();

  virtual void Reshape() { LOG(INFO) << "Reshape Operator!"; }
  virtual void Forward() { LOG(INFO) << "Forward Operator!"; }

  bool has_argument(const std::string &name) const {
    return arg_helper_.HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string &name, const T &default_value) const {
    return arg_helper_.template GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  bool has_single_argument_of_type(const std::string &name) const {
    return arg_helper_.template HasSingleArgumentOfType<T>(name);
  }
  template <typename T>
  const std::vector<T> get_repeated_argument(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    return arg_helper_.template GetRepeatedArgument<T>(name, default_value);
  }

  const std::string &name() const { return op_name_; }
  void set_name(const std::string &name) { op_name_ = name; }

  const std::string &type() const { return op_type_; }
  void set_type(const std::string &type) { op_type_ = type; }

  template <typename T>
  const Blob<T> *bottoms(int i) const {
    CHECK(check_index(i, bottoms_size()));
    return op_ws_->GetBlob<T>(bottom_names_[i]);
  }
  template <typename T>
  Blob<T> *mutable_bottoms(int i) {
    return const_cast<Blob<T> *>(
        static_cast<const Operator *>(this)->bottoms<T>(i));
  }
  template <typename T>
  void add_bottoms(const std::string &bottom_name) {
    op_ws_->CreateBlob<T>(bottom_name);
    bottom_names_.push_back(bottom_name);
  }
  template <typename T>
  void set_bottoms(int i, int count, const T *data) {
    CHECK(check_index(i, bottoms_size()));
    mutable_bottoms<T>(i)->set_data(data, count);
  }
  const std::string &bottoms_name(int i) const {
    CHECK(check_index(i, bottoms_size()));
    return bottom_names_[i];
  }
  const std::string bottoms_type(int i) const {
    CHECK(check_index(i, bottoms_size()));
    return op_ws_->GetBlobType(bottom_names_[i]);
  }
  int bottoms_size() const { return static_cast<int>(bottom_names_.size()); }

  template <typename T>
  const Blob<T> *tops(int i) const {
    CHECK(check_index(i, tops_size()));
    return op_ws_->GetBlob<T>(top_names_[i]);
  }
  template <typename T>
  Blob<T> *mutable_tops(int i) {
    return const_cast<Blob<T> *>(
        static_cast<const Operator *>(this)->tops<T>(i));
  }
  template <typename T>
  void add_tops(const std::string &top_name) {
    op_ws_->CreateBlob<T>(top_name);
    top_names_.push_back(top_name);
  }
  template <typename T>
  void set_tops(int i, int count, const T *data) {
    CHECK(check_index(i, tops_size()));
    mutable_tops<T>(i)->set_data(data, count);
  }
  const std::string tops_name(int i) const {
    CHECK(check_index(i, tops_size()));
    return top_names_[i];
  }
  const std::string tops_type(int i) const {
    CHECK(check_index(i, tops_size()));
    return op_ws_->GetBlobType(top_names_[i]);
  }
  int tops_size() const { return static_cast<int>(top_names_.size()); }

  template <typename T>
  const Blob<T> *blobs(int i) const {
    CHECK(check_index(i, blobs_size()));
    return op_ws_->GetBlob<T>(blob_names_[i]);
  }
  template <typename T>
  Blob<T> *mutable_blobs(int i) {
    return const_cast<Blob<T> *>(
        static_cast<const Operator *>(this)->blobs<T>(i));
  }
  template <typename T>
  void add_blobs(const std::string &blob_name) {
    op_ws_->CreateBlob<T>(blob_name);
    blob_names_.push_back(blob_name);
  }
  template <typename T>
  void set_blobs(int i, int count, const T *data) {
    CHECK(check_index(i, blobs_size()));
    mutable_blobs<T>(i)->set_data(data, count);
  }
  const std::string blobs_name(int i) const {
    CHECK(check_index(i, blobs_size()));
    return blob_names_[i];
  }
  const std::string blobs_type(int i) const {
    CHECK(check_index(i, blobs_size()));
    return op_ws_->GetBlobType(blob_names_[i]);
  }
  int blobs_size() const { return static_cast<int>(blob_names_.size()); }

 protected:
  std::string op_name_, op_type_;
  Workspace *op_ws_ = nullptr;

 private:
  bool check_index(int i, int size) const { return i >= 0 && i < size; }

  shadow::OpParam op_param_;
  ArgumentHelper arg_helper_;

  VecString bottom_names_, top_names_, blob_names_;

  DISABLE_COPY_AND_ASSIGN(Operator);
};

using VecOp = std::vector<Operator *>;

Operator *CreateOperator(const shadow::OpParam &op_param, Workspace *ws);

SHADOW_DECLARE_REGISTRY(OperatorRegistry, Operator, const shadow::OpParam &,
                        Workspace *);
#define REGISTER_OPERATOR(name, ...) \
  SHADOW_REGISTER_CLASS(OperatorRegistry, name, __VA_ARGS__)

class StaticLinkingProtector {
 public:
  StaticLinkingProtector() {
    const auto registered_ops = OperatorRegistry()->Keys().size();
    if (registered_ops == 0) {
      LOG(FATAL) << "You might have made a build error: the Shadow library "
                    "does not seem to be linked with whole-static library "
                    "option. To do so, use -Wl,-force_load (clang) or "
                    "-Wl,--whole-archive (gcc) to link the Shadow library.";
    }
  }
};

}  // namespace Shadow

#endif  // SHADOW_CORE_OPERATOR_HPP
