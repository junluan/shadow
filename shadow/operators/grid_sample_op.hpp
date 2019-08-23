#ifndef SHADOW_OPERATORS_GRID_SAMPLE_OP_HPP
#define SHADOW_OPERATORS_GRID_SAMPLE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class GridSampleOp : public Operator {
 public:
  GridSampleOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    mode_ = get_single_argument<int>("mode", 1);
    padding_mode_ = get_single_argument<int>("padding_mode", 0);
    CHECK(padding_mode_ == 0 || padding_mode_ == 1);

#if defined(USE_CUDNN)
    use_cudnn_ = mode_ == 1 && padding_mode_ == 0;
    if (use_cudnn_) {
      cudnn::createSpatialTransformerDesc<float>(&spatial_transformer_desc_);
      cudnn::createTensorDesc<float>(&bottom_desc_);
      cudnn::createTensorDesc<float>(&top_desc_);
    }
#endif
  }
  ~GridSampleOp() override {
#if defined(USE_CUDNN)
    if (spatial_transformer_desc_ != nullptr) {
      cudnnDestroySpatialTransformerDescriptor(spatial_transformer_desc_);
      spatial_transformer_desc_ = nullptr;
    }
    if (bottom_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_desc_);
      bottom_desc_ = nullptr;
    }
    if (top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(top_desc_);
      top_desc_ = nullptr;
    }
#endif
  }

  void Forward() override;

 private:
  int mode_, padding_mode_;
  bool use_cudnn_ = false;

#if defined(USE_CUDNN)
  cudnnSpatialTransformerDescriptor_t spatial_transformer_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
#endif
};

namespace Vision {

template <typename T>
void GridSample(const T* in_data, const VecInt& in_shape,
                const float* grid_data, int mode, int padding_mode,
                const VecInt& out_shape, T* out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_GRID_SAMPLE_OP_HPP
