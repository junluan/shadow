#ifndef SHADOW_CORE_PARAMS_HPP
#define SHADOW_CORE_PARAMS_HPP

#if defined(USE_Protobuf)
#include "proto/shadow.pb.h"
#endif

#include <string>
#include <vector>

namespace Shadow {

#if !defined(USE_Protobuf)
namespace shadow {

#define REPEATED_FIELD_FUNC(NAME, TYPE)                                     \
  const std::vector<TYPE> &NAME() const { return NAME##_; }                 \
  std::vector<TYPE> *mutable_##NAME() { return &NAME##_; }                  \
  void set_##NAME(int index, const TYPE &value) { NAME##_[index] = value; } \
  void add_##NAME() { NAME##_.resize(NAME##_.size() + 1); }                 \
  void add_##NAME(const TYPE &value) { NAME##_.push_back(value); }          \
  const TYPE &NAME(int index) const { return NAME##_[index]; }              \
  TYPE *mutable_##NAME(int index) { return &NAME##_[index]; }               \
  int NAME##_size() const { return NAME##_.size(); }                        \
  void clear_##NAME() { NAME##_.clear(); }

#define OPTIONAL_FIELD_DEFAULT_FUNC(NAME, TYPE, DEFAULT)  \
  const TYPE &NAME() const { return NAME##_; }            \
  void set_##NAME(const TYPE &value) { NAME##_ = value; } \
  void clear_##NAME() { NAME##_ = DEFAULT; }

#define OPTIONAL_NESTED_MESSAGE_FUNC(NAME, TYPE)              \
  const TYPE &NAME() const {                                  \
    return NAME##_ != nullptr ? *NAME##_ : default_##NAME##_; \
  }                                                           \
  TYPE *mutable_##NAME() {                                    \
    if (NAME##_ == nullptr) {                                 \
      NAME##_ = new TYPE;                                     \
    }                                                         \
    return NAME##_;                                           \
  }                                                           \
  void clear_##NAME() {                                       \
    if (NAME##_ != nullptr) {                                 \
      delete NAME##_;                                         \
      NAME##_ = nullptr;                                      \
    }                                                         \
  }

#define EQUAL_OPERATOR_FUNC(NAME)     \
  NAME &operator=(const NAME &from) { \
    CopyFrom(from);                   \
    return *this;                     \
  }

#define DEFAULT_CONSTRUCTOR_FUNC(NAME)     \
  NAME() {}                                \
  NAME(const NAME &from) { *this = from; } \
  ~NAME() { Clear(); }

#define DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(NAME) \
  DEFAULT_CONSTRUCTOR_FUNC(NAME)                           \
  EQUAL_OPERATOR_FUNC(NAME)

class BlobShape {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(BlobShape);

  void CopyFrom(const BlobShape &from) {
    if (&from == this) return;
    Clear();
    dim_ = from.dim_;
  }

  REPEATED_FIELD_FUNC(dim, int);

  void Clear() { clear_dim(); }

 private:
  std::vector<int> dim_;
};

class Blob {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(Blob);

  void CopyFrom(const Blob &from) {
    if (&from == this) return;
    Clear();
    data_ = from.data_;
    *mutable_shape() = from.shape();
  }

  REPEATED_FIELD_FUNC(data, float);
  OPTIONAL_NESTED_MESSAGE_FUNC(shape, BlobShape);

  void Clear() {
    clear_data();
    clear_shape();
  }

 private:
  std::vector<float> data_;
  BlobShape *shape_ = nullptr, default_shape_;
};

enum ActivateParam_ActivateType {
  ActivateParam_ActivateType_Linear = 0,
  ActivateParam_ActivateType_Relu = 1,
  ActivateParam_ActivateType_Leaky = 2,
  ActivateParam_ActivateType_PRelu = 3
};

class ActivateParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(ActivateParam);

  void CopyFrom(const ActivateParam &from) {
    if (&from == this) return;
    Clear();
    type_ = from.type_;
    channel_shared_ = from.channel_shared_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(type, ActivateParam_ActivateType,
                              ActivateParam_ActivateType_Relu);
  OPTIONAL_FIELD_DEFAULT_FUNC(channel_shared, bool, false);

  void Clear() {
    clear_type();
    clear_channel_shared();
  }

 private:
  ActivateParam_ActivateType type_ = ActivateParam_ActivateType_Relu;
  bool channel_shared_ = false;
};

class BatchNormParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(BatchNormParam);

  void CopyFrom(const BatchNormParam &from) {
    if (&from == this) return;
    Clear();
    use_global_stats_ = from.use_global_stats_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(use_global_stats, bool, true);

  void Clear() { clear_use_global_stats(); }

 private:
  bool use_global_stats_ = true;
};

class BiasParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(BiasParam);

  void CopyFrom(const BiasParam &from) {
    if (&from == this) return;
    Clear();
    axis_ = from.axis_;
    num_axes_ = from.num_axes_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(num_axes, int, 1);

  void Clear() {
    clear_axis();
    clear_num_axes();
  }

 private:
  int axis_ = 1, num_axes_ = 1;
};

class ConcatParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(ConcatParam);

  void CopyFrom(const ConcatParam &from) {
    if (&from == this) return;
    Clear();
    axis_ = from.axis_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);

  void Clear() { clear_axis(); }

 private:
  int axis_ = 1;
};

class ConnectedParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(ConnectedParam);

  void CopyFrom(const ConnectedParam &from) {
    if (&from == this) return;
    Clear();
    num_output_ = from.num_output_;
    bias_term_ = from.bias_term_;
    transpose_ = from.transpose_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(num_output, int, -1);
  bool has_num_output() const { return num_output_ > 0; }

  OPTIONAL_FIELD_DEFAULT_FUNC(bias_term, bool, true);
  OPTIONAL_FIELD_DEFAULT_FUNC(transpose, bool, false);

  void Clear() {
    clear_num_output();
    clear_bias_term();
    clear_transpose();
  }

 private:
  int num_output_ = -1;
  bool bias_term_ = true, transpose_ = false;
};

class ConvolutionParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(ConvolutionParam);

  void CopyFrom(const ConvolutionParam &from) {
    if (&from == this) return;
    Clear();
    num_output_ = from.num_output_;
    kernel_size_ = from.kernel_size_;
    stride_ = from.stride_;
    pad_ = from.pad_;
    group_ = from.group_;
    dilation_ = from.dilation_;
    bias_term_ = from.bias_term_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(num_output, int, -1);
  bool has_num_output() const { return num_output_ > 0; }

  OPTIONAL_FIELD_DEFAULT_FUNC(kernel_size, int, -1);
  bool has_kernel_size() const { return kernel_size_ > 0; }

  OPTIONAL_FIELD_DEFAULT_FUNC(stride, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(pad, int, 0);
  OPTIONAL_FIELD_DEFAULT_FUNC(dilation, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(group, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(bias_term, bool, true);

  void Clear() {
    clear_num_output();
    clear_kernel_size();
    clear_stride();
    clear_pad();
    clear_dilation();
    clear_bias_term();
  }

 private:
  int num_output_ = -1, kernel_size_ = -1, stride_ = 1, pad_ = 0, group_ = 1,
      dilation_ = 1;
  bool bias_term_ = true;
};

class DataParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(DataParam);

  void CopyFrom(const DataParam &from) {
    if (&from == this) return;
    Clear();
    *mutable_data_shape() = from.data_shape();
    scale_ = from.scale_;
    mean_value_ = from.mean_value_;
  }

  OPTIONAL_NESTED_MESSAGE_FUNC(data_shape, BlobShape);
  OPTIONAL_FIELD_DEFAULT_FUNC(scale, float, 1);
  REPEATED_FIELD_FUNC(mean_value, float)

  void Clear() {
    clear_data_shape();
    clear_scale();
    clear_mean_value();
  }

 private:
  BlobShape *data_shape_ = nullptr, default_data_shape_;
  float scale_ = 1;
  std::vector<float> mean_value_;
};

enum EltwiseParam_EltwiseOp {
  EltwiseParam_EltwiseOp_Prod = 0,
  EltwiseParam_EltwiseOp_Sum = 1,
  EltwiseParam_EltwiseOp_Max = 2
};

class EltwiseParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(EltwiseParam);

  void CopyFrom(const EltwiseParam &from) {
    if (&from == this) return;
    Clear();
    operation_ = from.operation_;
    coeff_ = from.coeff_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(operation, EltwiseParam_EltwiseOp,
                              EltwiseParam_EltwiseOp_Sum);
  REPEATED_FIELD_FUNC(coeff, float);

  void Clear() {
    clear_operation();
    clear_coeff();
  }

 private:
  EltwiseParam_EltwiseOp operation_ = EltwiseParam_EltwiseOp_Sum;
  std::vector<float> coeff_;
};

class FlattenParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(FlattenParam);

  void CopyFrom(const FlattenParam &from) {
    if (&from == this) return;
    Clear();
    axis_ = from.axis_;
    end_axis_ = from.end_axis_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(end_axis, int, -1);

  void Clear() {
    clear_axis();
    clear_end_axis();
  }

 private:
  int axis_ = 1, end_axis_ = -1;
};

enum LRNParam_NormRegion {
  LRNParam_NormRegion_AcrossChannels = 0,
  LRNParam_NormRegion_WithinChannel = 1
};

class LRNParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(LRNParam);

  void CopyFrom(const LRNParam &from) {
    if (&from == this) return;
    Clear();
    norm_region_ = from.norm_region_;
    local_size_ = from.local_size_;
    alpha_ = from.alpha_;
    beta_ = from.beta_;
    k_ = from.k_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(local_size, int, 5);
  OPTIONAL_FIELD_DEFAULT_FUNC(alpha, float, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(beta, float, 0.75);
  OPTIONAL_FIELD_DEFAULT_FUNC(norm_region, LRNParam_NormRegion,
                              LRNParam_NormRegion_AcrossChannels);
  OPTIONAL_FIELD_DEFAULT_FUNC(k, float, 1);

  void Clear() {
    clear_local_size();
    clear_alpha();
    clear_beta();
    clear_norm_region();
    clear_k();
  }

 private:
  LRNParam_NormRegion norm_region_ = LRNParam_NormRegion_AcrossChannels;
  int local_size_ = 5;
  float alpha_ = 1, beta_ = 0.75, k_ = 1;
};

class NormalizeParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(NormalizeParam);

  void CopyFrom(const NormalizeParam &from) {
    if (&from == this) return;
    Clear();
    across_spatial_ = from.across_spatial_;
    channel_shared_ = from.channel_shared_;
    scale_ = from.scale_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(across_spatial, bool, true);
  OPTIONAL_FIELD_DEFAULT_FUNC(channel_shared, bool, true);
  REPEATED_FIELD_FUNC(scale, float);

  void Clear() {
    clear_across_spatial();
    clear_channel_shared();
    clear_scale();
  }

 private:
  bool across_spatial_ = true, channel_shared_ = true;
  std::vector<float> scale_;
};

class PermuteParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(PermuteParam);

  void CopyFrom(const PermuteParam &from) {
    if (&from == this) return;
    Clear();
    order_ = from.order_;
  }

  REPEATED_FIELD_FUNC(order, int);

  void Clear() { clear_order(); }

 private:
  std::vector<int> order_;
};

enum PoolingParam_PoolType {
  PoolingParam_PoolType_Max = 0,
  PoolingParam_PoolType_Ave = 1
};

class PoolingParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(PoolingParam);

  void CopyFrom(const PoolingParam &from) {
    if (&from == this) return;
    Clear();
    pool_ = from.pool_;
    kernel_size_ = from.kernel_size_;
    stride_ = from.stride_;
    pad_ = from.pad_;
    global_pooling_ = from.global_pooling_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(pool, PoolingParam_PoolType,
                              PoolingParam_PoolType_Max);
  OPTIONAL_FIELD_DEFAULT_FUNC(stride, int, 1)
  OPTIONAL_FIELD_DEFAULT_FUNC(pad, int, 0)
  OPTIONAL_FIELD_DEFAULT_FUNC(global_pooling, bool, false)

  OPTIONAL_FIELD_DEFAULT_FUNC(kernel_size, int, -1);
  bool has_kernel_size() const { return kernel_size_ > 0; }

  void Clear() {
    clear_pool();
    clear_kernel_size();
    clear_stride();
    clear_pad();
    clear_global_pooling();
  }

 private:
  PoolingParam_PoolType pool_ = PoolingParam_PoolType_Max;
  int kernel_size_ = -1, stride_ = 1, pad_ = 0;
  bool global_pooling_ = false;
};

class PriorBoxParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(PriorBoxParam);

  void CopyFrom(const PriorBoxParam &from) {
    if (&from == this) return;
    Clear();
    min_size_ = from.min_size_;
    max_size_ = from.max_size_;
    aspect_ratio_ = from.aspect_ratio_;
    variance_ = from.variance_;
    flip_ = from.flip_;
    clip_ = from.clip_;
    step_ = from.step_;
    offset_ = from.offset_;
  }

  REPEATED_FIELD_FUNC(min_size, float);
  REPEATED_FIELD_FUNC(max_size, float);
  REPEATED_FIELD_FUNC(aspect_ratio, float);
  OPTIONAL_FIELD_DEFAULT_FUNC(flip, bool, true);
  OPTIONAL_FIELD_DEFAULT_FUNC(clip, bool, false);
  REPEATED_FIELD_FUNC(variance, float);

  OPTIONAL_FIELD_DEFAULT_FUNC(step, float, 0);
  bool has_step() const { return step_ > 0; }

  OPTIONAL_FIELD_DEFAULT_FUNC(offset, float, 0.5);

  void Clear() {
    clear_min_size();
    clear_max_size();
    clear_aspect_ratio();
    clear_flip();
    clear_clip();
    clear_variance();
  }

 private:
  std::vector<float> min_size_, max_size_, aspect_ratio_, variance_;
  bool flip_ = true, clip_ = false;
  float step_ = -1, offset_ = 0.5;
};

class ReorgParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(ReorgParam);

  void CopyFrom(const ReorgParam &from) {
    if (&from == this) return;
    Clear();
    stride_ = from.stride_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(stride, int, 2);

  void Clear() { clear_stride(); }

 private:
  int stride_ = 2;
};

class ReshapeParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(ReshapeParam);

  void CopyFrom(const ReshapeParam &from) {
    if (&from == this) return;
    Clear();
    *mutable_shape() = from.shape();
    axis_ = from.axis_;
    num_axes_ = from.num_axes_;
  }

  OPTIONAL_NESTED_MESSAGE_FUNC(shape, BlobShape);
  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 0);
  OPTIONAL_FIELD_DEFAULT_FUNC(num_axes, int, -1);

  void Clear() {
    clear_shape();
    clear_axis();
    clear_num_axes();
  }

 private:
  BlobShape *shape_ = nullptr, default_shape_;
  int axis_ = 0, num_axes_ = -1;
};

class ScaleParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(ScaleParam);

  void CopyFrom(const ScaleParam &from) {
    if (&from == this) return;
    Clear();
    axis_ = from.axis_;
    num_axes_ = from.num_axes_;
    bias_term_ = from.bias_term_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(num_axes, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(bias_term, bool, false);

  void Clear() {
    clear_axis();
    clear_num_axes();
    clear_bias_term();
  }

 private:
  int axis_ = 1, num_axes_ = 1;
  bool bias_term_ = false;
};

class SoftmaxParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(SoftmaxParam);

  void CopyFrom(const SoftmaxParam &from) {
    if (&from == this) return;
    Clear();
    axis_ = from.axis_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);

  void Clear() { clear_axis(); }

 private:
  int axis_ = 1;
};

class OpParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(OpParam);

  void CopyFrom(const OpParam &from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    type_ = from.type_;
    bottom_ = from.bottom_;
    top_ = from.top_;
    blobs_ = from.blobs_;
    *mutable_activate_param() = from.activate_param();
    *mutable_batch_norm_param() = from.batch_norm_param();
    *mutable_bias_param() = from.bias_param();
    *mutable_concat_param() = from.concat_param();
    *mutable_connected_param() = from.connected_param();
    *mutable_convolution_param() = from.convolution_param();
    *mutable_data_param() = from.data_param();
    *mutable_eltwise_param() = from.eltwise_param();
    *mutable_flatten_param() = from.flatten_param();
    *mutable_lrn_param() = from.lrn_param();
    *mutable_normalize_param() = from.normalize_param();
    *mutable_permute_param() = from.permute_param();
    *mutable_pooling_param() = from.pooling_param();
    *mutable_prior_box_param() = from.prior_box_param();
    *mutable_reorg_param() = from.reorg_param();
    *mutable_reshape_param() = from.reshape_param();
    *mutable_scale_param() = from.scale_param();
    *mutable_softmax_param() = from.softmax_param();
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(name, std::string, "");
  OPTIONAL_FIELD_DEFAULT_FUNC(type, std::string, "");

  REPEATED_FIELD_FUNC(bottom, std::string);
  REPEATED_FIELD_FUNC(top, std::string);
  REPEATED_FIELD_FUNC(blobs, Blob);

  OPTIONAL_NESTED_MESSAGE_FUNC(activate_param, ActivateParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(batch_norm_param, BatchNormParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(bias_param, BiasParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(concat_param, ConcatParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(connected_param, ConnectedParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(convolution_param, ConvolutionParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(data_param, DataParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(eltwise_param, EltwiseParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(flatten_param, FlattenParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(lrn_param, LRNParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(normalize_param, NormalizeParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(permute_param, PermuteParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(pooling_param, PoolingParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(prior_box_param, PriorBoxParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(reorg_param, ReorgParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(reshape_param, ReshapeParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(scale_param, ScaleParam);
  OPTIONAL_NESTED_MESSAGE_FUNC(softmax_param, SoftmaxParam);

  void Clear() {
    clear_name();
    clear_type();
    clear_bottom();
    clear_top();
    clear_blobs();
    clear_activate_param();
    clear_batch_norm_param();
    clear_bias_param();
    clear_concat_param();
    clear_connected_param();
    clear_convolution_param();
    clear_data_param();
    clear_eltwise_param();
    clear_flatten_param();
    clear_lrn_param();
    clear_normalize_param();
    clear_permute_param();
    clear_pooling_param();
    clear_prior_box_param();
    clear_reorg_param();
    clear_reshape_param();
    clear_scale_param();
    clear_softmax_param();
  }

 private:
  std::string name_, type_;
  std::vector<std::string> bottom_, top_;
  std::vector<Blob> blobs_;

  ActivateParam *activate_param_ = nullptr, default_activate_param_;
  BatchNormParam *batch_norm_param_ = nullptr, default_batch_norm_param_;
  BiasParam *bias_param_ = nullptr, default_bias_param_;
  ConcatParam *concat_param_ = nullptr, default_concat_param_;
  ConnectedParam *connected_param_ = nullptr, default_connected_param_;
  ConvolutionParam *convolution_param_ = nullptr, default_convolution_param_;
  DataParam *data_param_ = nullptr, default_data_param_;
  EltwiseParam *eltwise_param_ = nullptr, default_eltwise_param_;
  FlattenParam *flatten_param_ = nullptr, default_flatten_param_;
  LRNParam *lrn_param_ = nullptr, default_lrn_param_;
  NormalizeParam *normalize_param_ = nullptr, default_normalize_param_;
  PermuteParam *permute_param_ = nullptr, default_permute_param_;
  PoolingParam *pooling_param_ = nullptr, default_pooling_param_;
  PriorBoxParam *prior_box_param_ = nullptr, default_prior_box_param_;
  ReorgParam *reorg_param_ = nullptr, default_reorg_param_;
  ReshapeParam *reshape_param_ = nullptr, default_reshape_param_;
  ScaleParam *scale_param_ = nullptr, default_scale_param_;
  SoftmaxParam *softmax_param_ = nullptr, default_softmax_param_;
};

class NetParam {
 public:
  DEFAULT_CONSTRUCTOR_WITH_EQUAL_OPERATOR_FUNC(NetParam);

  void CopyFrom(const NetParam &from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    op_ = from.op_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(name, std::string, "");
  REPEATED_FIELD_FUNC(op, OpParam);

  void Clear() {
    clear_name();
    clear_op();
  }

 private:
  std::string name_;
  std::vector<OpParam> op_;
};

}  // namespace shadow
#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_PARAMS_HPP
