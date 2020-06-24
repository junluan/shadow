#ifndef SHADOW_CORE_PARAMS_HPP_
#define SHADOW_CORE_PARAMS_HPP_

#include "config.hpp"

#if defined(USE_Protobuf)
#include "proto/shadow.pb.h"
#endif

#include <limits>
#include <string>
#include <vector>

#if !defined(USE_Protobuf)

namespace shadow {

#define REPEATED_FIELD_FUNC(NAME, TYPE)                                     \
  const std::vector<TYPE>& NAME() const { return NAME##_; }                 \
  std::vector<TYPE>* mutable_##NAME() { return &NAME##_; }                  \
  void set_##NAME(int index, const TYPE& value) { NAME##_[index] = value; } \
  TYPE* add_##NAME() {                                                      \
    NAME##_.resize(NAME##_.size() + 1);                                     \
    return &NAME##_[NAME##_.size() - 1];                                    \
  }                                                                         \
  void add_##NAME(const TYPE& value) { NAME##_.push_back(value); }          \
  const TYPE& NAME(int index) const { return NAME##_[index]; }              \
  TYPE* mutable_##NAME(int index) { return &NAME##_[index]; }               \
  int NAME##_size() const { return static_cast<int>(NAME##_.size()); }      \
  bool has_##NAME() const { return !NAME##_.empty(); }                      \
  void clear_##NAME() { NAME##_.clear(); }

#define OPTIONAL_FIELD_FUNC(NAME, TYPE, DEFAULT)    \
  const TYPE& NAME() const { return NAME##_; }      \
  void set_##NAME(const TYPE& value) {              \
    NAME##_ = value;                                \
    has_##NAME##_ = true;                           \
  }                                                 \
  bool has_##NAME() const { return has_##NAME##_; } \
  void clear_##NAME() {                             \
    NAME##_ = DEFAULT;                              \
    has_##NAME##_ = false;                          \
  }

#define DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(NAME) \
  NAME() = default;                                        \
  NAME(const NAME& from) { *this = from; }                 \
  ~NAME() { Clear(); }                                     \
  NAME& operator=(const NAME& from) {                      \
    CopyFrom(from);                                        \
    return *this;                                          \
  }

class Blob {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(Blob);

  void CopyFrom(const Blob& from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    type_ = from.type_;
    shape_ = from.shape_;
    data_f_ = from.data_f_;
    data_i_ = from.data_i_;
    data_b_ = from.data_b_;
    has_name_ = from.has_name_;
    has_type_ = from.has_type_;
  }

  OPTIONAL_FIELD_FUNC(name, std::string, "");
  OPTIONAL_FIELD_FUNC(type, std::string, "");
  REPEATED_FIELD_FUNC(shape, int);
  REPEATED_FIELD_FUNC(data_f, float);
  REPEATED_FIELD_FUNC(data_i, int);
  REPEATED_FIELD_FUNC(data_b, std::vector<char>);

  void Clear() {
    clear_name();
    clear_type();
    clear_shape();
    clear_data_f();
    clear_data_i();
    clear_data_b();
  }

 private:
  std::string name_{"None"}, type_{"None"};
  std::vector<int> shape_;
  std::vector<float> data_f_;
  std::vector<int> data_i_;
  std::vector<std::vector<char>> data_b_;
  bool has_name_{false}, has_type_{false};
};

class Argument {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(Argument);

  void CopyFrom(const Argument& from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    s_f_ = from.s_f_;
    s_i_ = from.s_i_;
    s_s_ = from.s_s_;
    v_f_ = from.v_f_;
    v_i_ = from.v_i_;
    v_s_ = from.v_s_;
    has_name_ = from.has_name_;
    has_s_f_ = from.has_s_f_;
    has_s_i_ = from.has_s_i_;
    has_s_s_ = from.has_s_s_;
  }

  OPTIONAL_FIELD_FUNC(name, std::string, "");
  OPTIONAL_FIELD_FUNC(s_f, float, std::numeric_limits<float>::max());
  OPTIONAL_FIELD_FUNC(s_i, int, std::numeric_limits<int>::max());
  OPTIONAL_FIELD_FUNC(s_s, std::string, "");
  REPEATED_FIELD_FUNC(v_f, float);
  REPEATED_FIELD_FUNC(v_i, int);
  REPEATED_FIELD_FUNC(v_s, std::string);

  void Clear() {
    clear_name();
    clear_s_f();
    clear_s_i();
    clear_s_s();
    clear_v_f();
    clear_v_i();
    clear_v_s();
  }

 private:
  std::string name_{"None"};
  float s_f_;
  int s_i_;
  std::string s_s_{"None"};
  std::vector<float> v_f_;
  std::vector<int> v_i_;
  std::vector<std::string> v_s_;
  bool has_name_{false}, has_s_f_{false}, has_s_i_{false}, has_s_s_{false};
};

class OpParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(OpParam);

  void CopyFrom(const OpParam& from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    type_ = from.type_;
    bottom_ = from.bottom_;
    top_ = from.top_;
    arg_ = from.arg_;
    has_name_ = from.has_name_;
    has_type_ = from.has_type_;
  }

  OPTIONAL_FIELD_FUNC(name, std::string, "");
  OPTIONAL_FIELD_FUNC(type, std::string, "");

  REPEATED_FIELD_FUNC(bottom, std::string);
  REPEATED_FIELD_FUNC(top, std::string);
  REPEATED_FIELD_FUNC(arg, Argument);

  void Clear() {
    clear_name();
    clear_type();
    clear_bottom();
    clear_top();
    clear_arg();
  }

 private:
  std::string name_{"None"}, type_{"None"};
  std::vector<std::string> bottom_, top_;
  std::vector<Argument> arg_;
  bool has_name_{false}, has_type_{false};
};

class NetParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(NetParam);

  void CopyFrom(const NetParam& from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    blob_ = from.blob_;
    op_ = from.op_;
    arg_ = from.arg_;
    has_name_ = from.has_name_;
  }

  OPTIONAL_FIELD_FUNC(name, std::string, "");
  REPEATED_FIELD_FUNC(blob, Blob);
  REPEATED_FIELD_FUNC(op, OpParam);
  REPEATED_FIELD_FUNC(arg, Argument);

  void Clear() {
    clear_name();
    clear_blob();
    clear_op();
    clear_arg();
  }

 private:
  std::string name_{"None"};
  std::vector<Blob> blob_;
  std::vector<OpParam> op_;
  std::vector<Argument> arg_;
  bool has_name_{false};
};

class MetaNetParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(MetaNetParam);

  void CopyFrom(const MetaNetParam& from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    network_ = from.network_;
    arg_ = from.arg_;
    has_name_ = from.has_name_;
  }

  OPTIONAL_FIELD_FUNC(name, std::string, "");
  REPEATED_FIELD_FUNC(network, NetParam);
  REPEATED_FIELD_FUNC(arg, Argument);

  void Clear() {
    clear_name();
    clear_network();
    clear_arg();
  }

 private:
  std::string name_{"None"};
  std::vector<NetParam> network_;
  std::vector<Argument> arg_;
  bool has_name_{false};
};

}  // namespace shadow

#endif

#endif  // SHADOW_CORE_PARAMS_HPP_
