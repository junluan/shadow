#ifndef SHADOW_PARAMS_HPP
#define SHADOW_PARAMS_HPP

#include <string>
#include <vector>

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

class BlobShape {
 public:
  BlobShape() {}
  ~BlobShape() { Clear(); }

  REPEATED_FIELD_FUNC(dim, int);

  void Clear() { clear_dim(); }

 private:
  std::vector<int> dim_;
};

class Blob {
 public:
  Blob() {}
  Blob(const Blob &from) { *this = from; }
  ~Blob() { Clear(); }

  Blob &operator=(const Blob &from) {
    CopyFrom(from);
    return *this;
  }

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

enum ActivateParameter_ActivateType {
  ActivateParameter_ActivateType_Linear = 0,
  ActivateParameter_ActivateType_Relu = 1,
  ActivateParameter_ActivateType_Leaky = 2
};

class ActivateParameter {
 public:
  ActivateParameter() {}
  ~ActivateParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(type, ActivateParameter_ActivateType,
                              ActivateParameter_ActivateType_Relu);

  void Clear() { clear_type(); }

 private:
  ActivateParameter_ActivateType type_ = ActivateParameter_ActivateType_Relu;
};

class BatchNormParameter {
 public:
  BatchNormParameter() {}
  ~BatchNormParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(use_global_stats, bool, true);

  void Clear() { clear_use_global_stats(); }

 private:
  bool use_global_stats_ = true;
};

class BiasParameter {
 public:
  BiasParameter() {}
  ~BiasParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(num_axes, int, 1);

  void Clear() {
    clear_axis();
    clear_num_axes();
  }

 private:
  int axis_ = 1, num_axes_ = 1;
};

class ConcatParameter {
 public:
  ConcatParameter() {}
  ~ConcatParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);

  void Clear() { clear_axis(); }

 private:
  int axis_ = 1;
};

class ConnectedParameter {
 public:
  ConnectedParameter() {}
  ~ConnectedParameter() { Clear(); }

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

class ConvolutionParameter {
 public:
  ConvolutionParameter() {}
  ~ConvolutionParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(num_output, int, -1);
  bool has_num_output() const { return num_output_ > 0; }

  OPTIONAL_FIELD_DEFAULT_FUNC(kernel_size, int, -1);
  bool has_kernel_size() const { return kernel_size_ > 0; }

  OPTIONAL_FIELD_DEFAULT_FUNC(stride, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(pad, int, 0);
  OPTIONAL_FIELD_DEFAULT_FUNC(dilation, int, 1);
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
  int num_output_ = -1, kernel_size_ = -1, stride_ = 1, pad_ = 0, dilation_ = 1;
  bool bias_term_ = true;
};

class DataParameter {
 public:
  DataParameter() {}
  DataParameter(const DataParameter &from) { *this = from; }
  ~DataParameter() { Clear(); }

  DataParameter &operator=(const DataParameter &from) {
    CopyFrom(from);
    return *this;
  }

  void CopyFrom(const DataParameter &from) {
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

class FlattenParameter {
 public:
  FlattenParameter() {}
  ~FlattenParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(end_axis, int, -1);

  void Clear() {
    clear_axis();
    clear_end_axis();
  }

 private:
  int axis_ = 1, end_axis_ = -1;
};

enum LRNParameter_NormRegion {
  LRNParameter_NormRegion_AcrossChannels = 0,
  LRNParameter_NormRegion_WithinChannel = 1
};

class LRNParameter {
 public:
  LRNParameter() {}
  ~LRNParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(local_size, int, 5);
  OPTIONAL_FIELD_DEFAULT_FUNC(alpha, float, 1);
  OPTIONAL_FIELD_DEFAULT_FUNC(beta, float, 0.75);
  OPTIONAL_FIELD_DEFAULT_FUNC(norm_region, LRNParameter_NormRegion,
                              LRNParameter_NormRegion_AcrossChannels);
  OPTIONAL_FIELD_DEFAULT_FUNC(k, float, 1);

  void Clear() {
    clear_local_size();
    clear_alpha();
    clear_beta();
    clear_norm_region();
    clear_k();
  }

 private:
  LRNParameter_NormRegion norm_region_ = LRNParameter_NormRegion_AcrossChannels;
  int local_size_ = 5;
  float alpha_ = 1, beta_ = 0.75, k_ = 1;
};

class NormalizeParameter {
 public:
  NormalizeParameter() {}
  ~NormalizeParameter() { Clear(); }

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

class PermuteParameter {
 public:
  PermuteParameter() {}
  ~PermuteParameter() { Clear(); }

  REPEATED_FIELD_FUNC(order, int);

  void Clear() { clear_order(); }

 private:
  std::vector<int> order_;
};

enum PoolingParameter_PoolType {
  PoolingParameter_PoolType_Max = 0,
  PoolingParameter_PoolType_Ave = 1
};

class PoolingParameter {
 public:
  PoolingParameter() {}
  ~PoolingParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(pool, PoolingParameter_PoolType,
                              PoolingParameter_PoolType_Max);
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
  PoolingParameter_PoolType pool_ = PoolingParameter_PoolType_Max;
  int kernel_size_ = -1, stride_ = 1, pad_ = 0;
  bool global_pooling_ = false;
};

class PriorBoxParameter {
 public:
  PriorBoxParameter() {}
  ~PriorBoxParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(min_size, float, -1);
  bool has_min_size() const { return min_size_ > 0; }

  OPTIONAL_FIELD_DEFAULT_FUNC(max_size, float, -1);
  bool has_max_size() const { return max_size_ > 0; }

  REPEATED_FIELD_FUNC(aspect_ratio, float);
  OPTIONAL_FIELD_DEFAULT_FUNC(flip, bool, true);
  OPTIONAL_FIELD_DEFAULT_FUNC(clip, bool, true);
  REPEATED_FIELD_FUNC(variance, float);

  void Clear() {
    clear_min_size();
    clear_max_size();
    clear_aspect_ratio();
    clear_flip();
    clear_clip();
    clear_variance();
  }

 private:
  float min_size_ = -1, max_size_ = -1;
  std::vector<float> aspect_ratio_, variance_;
  bool flip_ = true, clip_ = true;
};

class ReorgParameter {
 public:
  ReorgParameter() {}
  ~ReorgParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(stride, int, 2);

  void Clear() { clear_stride(); }

 private:
  int stride_ = 2;
};

class ReshapeParameter {
 public:
  ReshapeParameter() {}
  ReshapeParameter(const ReshapeParameter &from) { *this = from; }
  ~ReshapeParameter() { Clear(); }

  ReshapeParameter &operator=(const ReshapeParameter &from) {
    CopyFrom(from);
    return *this;
  }

  void CopyFrom(const ReshapeParameter &from) {
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

class ScaleParameter {
 public:
  ScaleParameter() {}
  ~ScaleParameter() { Clear(); }

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

class SoftmaxParameter {
 public:
  SoftmaxParameter() {}
  ~SoftmaxParameter() { Clear(); }

  OPTIONAL_FIELD_DEFAULT_FUNC(axis, int, 1);

  void Clear() { clear_axis(); }

 private:
  int axis_ = 1;
};

class LayerParameter {
 public:
  LayerParameter() {}
  LayerParameter(const LayerParameter &from) { *this = from; }
  ~LayerParameter() { Clear(); }

  LayerParameter &operator=(const LayerParameter &from) {
    CopyFrom(from);
    return *this;
  }

  void CopyFrom(const LayerParameter &from) {
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

  OPTIONAL_NESTED_MESSAGE_FUNC(activate_param, ActivateParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(batch_norm_param, BatchNormParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(bias_param, BiasParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(concat_param, ConcatParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(connected_param, ConnectedParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(convolution_param, ConvolutionParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(data_param, DataParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(flatten_param, FlattenParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(lrn_param, LRNParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(normalize_param, NormalizeParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(permute_param, PermuteParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(pooling_param, PoolingParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(prior_box_param, PriorBoxParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(reorg_param, ReorgParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(reshape_param, ReshapeParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(scale_param, ScaleParameter);
  OPTIONAL_NESTED_MESSAGE_FUNC(softmax_param, SoftmaxParameter);

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

  ActivateParameter *activate_param_ = nullptr, default_activate_param_;
  BatchNormParameter *batch_norm_param_ = nullptr, default_batch_norm_param_;
  BiasParameter *bias_param_ = nullptr, default_bias_param_;
  ConcatParameter *concat_param_ = nullptr, default_concat_param_;
  ConnectedParameter *connected_param_ = nullptr, default_connected_param_;
  ConvolutionParameter *convolution_param_ = nullptr,
                       default_convolution_param_;
  DataParameter *data_param_ = nullptr, default_data_param_;
  FlattenParameter *flatten_param_ = nullptr, default_flatten_param_;
  LRNParameter *lrn_param_ = nullptr, default_lrn_param_;
  NormalizeParameter *normalize_param_ = nullptr, default_normalize_param_;
  PermuteParameter *permute_param_ = nullptr, default_permute_param_;
  PoolingParameter *pooling_param_ = nullptr, default_pooling_param_;
  PriorBoxParameter *prior_box_param_ = nullptr, default_prior_box_param_;
  ReorgParameter *reorg_param_ = nullptr, default_reorg_param_;
  ReshapeParameter *reshape_param_ = nullptr, default_reshape_param_;
  ScaleParameter *scale_param_ = nullptr, default_scale_param_;
  SoftmaxParameter *softmax_param_ = nullptr, default_softmax_param_;
};

class NetParameter {
 public:
  NetParameter() {}
  NetParameter(const NetParameter &from) { *this = from; }
  ~NetParameter() { Clear(); }

  NetParameter &operator=(const NetParameter &from) {
    CopyFrom(from);
    return *this;
  }

  void CopyFrom(const NetParameter &from) {
    if (&from == this) return;
    Clear();
    name_ = from.name_;
    layer_ = from.layer_;
  }

  OPTIONAL_FIELD_DEFAULT_FUNC(name, std::string, "");
  REPEATED_FIELD_FUNC(layer, LayerParameter);

  void Clear() {
    clear_name();
    clear_layer();
  }

 private:
  std::string name_;
  std::vector<LayerParameter> layer_;
};

}  // namespace shadow
#endif

#endif  // SHADOW_PARAMS_HPP
