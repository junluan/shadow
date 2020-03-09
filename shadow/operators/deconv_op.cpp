#include "deconv_op.hpp"

#include "activate_op.hpp"

namespace Shadow {

void DeconvOp::Forward() {
  if (bias_term_) {
    CHECK_EQ(bottoms_size(), 3);
  } else {
    CHECK_EQ(bottoms_size(), 2);
  }

  const auto bottom = bottoms(0);
  const auto weight = bottoms(1);
  auto top = tops(0);

  CHECK_NE(bottom, top);

  int batch = bottom->shape(0), in_c = bottom->shape(1),
      in_h = bottom->shape(2), in_w = bottom->shape(3);

  CHECK_EQ(in_c % group_, 0);

  auto top_shape = bottom->shape();
  top_shape[1] = num_output_;
  top_shape[2] =
      deconv_out_size(in_h, kernel_size_h_, stride_h_, pad_h_, dilation_);
  top_shape[3] =
      deconv_out_size(in_w, kernel_size_w_, stride_w_, pad_w_, dilation_);
  top->reshape(top_shape);

  conv_in_c = num_output_, conv_out_c = in_c;

  conv_out_spatial_dim_ = bottom->count(2);
  out_spatial_dim_ = top->count(2);
  kernel_dim_ = kernel_size_h_ * kernel_size_w_ * conv_in_c / group_;

  weight_offset_ = conv_out_c * kernel_dim_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_c * conv_out_spatial_dim_ / group_;

  Blas::Set(top->count(), 0, top->mutable_data<float>(), 0, ws_->Ctx());

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    cudnn::setConvolution2dDesc<float>(&conv_desc_, pad_h_, pad_w_, stride_h_,
                                       stride_w_, dilation_, dilation_, group_);
    cudnn::setTensor4dDesc<float>(&bottom_desc_, batch, in_c, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&top_desc_, batch, num_output_, top_shape[2],
                                  top_shape[3]);
    cudnn::setFilter4dDesc<float>(&filter_desc_, conv_out_c, conv_in_c / group_,
                                  kernel_size_h_, kernel_size_w_);
    if (bias_term_) {
      cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output_, 1, 1);
    }
    if (activate_type_ == 1) {
      cudnn::setActivationDesc<float>(&activate_desc_, activate_type_, 0);
    }

    size_t workspace_limit_bytes = group_ == 1 ? 64 * 1024 * 1024 : 0;

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t(ws_->Ctx()->cudnn_handle()), filter_desc_, bottom_desc_,
        conv_desc_, top_desc_,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_));

    size_t workspace_bwd_size = 0;

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle_t(ws_->Ctx()->cudnn_handle()), filter_desc_, bottom_desc_,
        conv_desc_, top_desc_, bwd_data_algo_, &workspace_bwd_size));

    std::shared_ptr<Blob> workspace = nullptr;
    const void *workspace_ptr = nullptr;
    if (workspace_bwd_size > 0) {
      ws_->GrowTempBuffer(workspace_bwd_size);
      workspace = ws_->CreateTempBlob({static_cast<int>(workspace_bwd_size)},
                                      DataType::kU8);
      workspace_ptr = workspace->data<unsigned char>();
    }

    CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnnHandle_t(ws_->Ctx()->cudnn_handle()), cudnn::dataType<float>::one,
        filter_desc_, weight->data<float>(), bottom_desc_,
        bottom->data<float>(), conv_desc_, bwd_data_algo_,
        const_cast<void *>(workspace_ptr), workspace_bwd_size,
        cudnn::dataType<float>::zero, top_desc_, top->mutable_data<float>()));
    if (bias_term_) {
      CUDNN_CHECK(cudnnAddTensor(
          cudnnHandle_t(ws_->Ctx()->cudnn_handle()),
          cudnn::dataType<float>::one, bias_desc_, bottoms(2)->data<float>(),
          cudnn::dataType<float>::one, top_desc_, top->mutable_data<float>()));
    }
    if (activate_type_ == 1) {
      CUDNN_CHECK(cudnnActivationForward(
          cudnnHandle_t(ws_->Ctx()->cudnn_handle()), activate_desc_,
          cudnn::dataType<float>::one, top_desc_, top->data<float>(),
          cudnn::dataType<float>::zero, top_desc_, top->mutable_data<float>()));
    }

    return;
  }
#endif

  int temp_count = kernel_dim_ * group_ * conv_out_spatial_dim_;
  if (bias_term_) {
    temp_count += out_spatial_dim_;
  }
  ws_->GrowTempBuffer(temp_count * sizeof(float));
  auto col_image = ws_->CreateTempBlob(
      {kernel_dim_ * group_, conv_out_spatial_dim_}, DataType::kF32);
  std::shared_ptr<Blob> biases_multiplier = nullptr;
  if (bias_term_) {
    biases_multiplier = ws_->CreateTempBlob({out_spatial_dim_}, DataType::kF32);
    Blas::Set(out_spatial_dim_, 1, biases_multiplier->mutable_data<float>(), 0,
              ws_->Ctx());
  }
  int top_num = top->num(), bottom_num = bottom->num();
  for (int b = 0; b < batch; ++b) {
    for (int g = 0; g < group_; ++g) {
      Blas::BlasSgemm(
          1, 0, kernel_dim_, conv_out_spatial_dim_, conv_out_c / group_, 1,
          weight->data<float>(), weight_offset_ * g, bottom->data<float>(),
          b * bottom_num + output_offset_ * g, 0,
          col_image->mutable_data<float>(), col_offset_ * g, ws_->Ctx());
    }
    Vision::Col2Im(col_image->data<float>(), top->shape(), b * top_num,
                   kernel_size_h_, kernel_size_w_, stride_h_, stride_w_, pad_h_,
                   pad_w_, dilation_, bottom->shape(),
                   top->mutable_data<float>(), ws_->Ctx());
    if (bias_term_) {
      Blas::BlasSgemm(0, 0, num_output_, out_spatial_dim_, 1, 1,
                      bottoms(2)->data<float>(), 0,
                      biases_multiplier->data<float>(), 0, 1,
                      top->mutable_data<float>(), b * top_num, ws_->Ctx());
    }
  }
  if (activate_type_ == 1) {
    Vision::Activate(top->data<float>(), top->mutable_data<float>(),
                     top->count(), activate_type_, 0, ws_->Ctx());
  }
}

REGISTER_OPERATOR(Deconv, DeconvOp);

namespace Vision {

#if !defined(USE_CUDA)
// check for 0 <= a < b
inline bool check_border(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void Col2Im(const T *col_data, const VecInt &in_shape, int offset,
            int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
            int pad_h, int pad_w, int dilation, const VecInt &out_shape,
            T *in_data, Context *context) {
  in_data += offset;
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int spatial_dim = in_h * in_w;
  for (int k_c = 0; k_c < in_c; ++k_c, in_data += spatial_dim) {
    for (int k_s = 0; k_s < kernel_size_h * kernel_size_w; ++k_s) {
      int k_h = k_s / kernel_size_w;
      int k_w = k_s % kernel_size_w;
      int im_row = -pad_h + k_h * dilation;
      for (int h = 0; h < out_h; ++h, im_row += stride_h) {
        if (check_border(im_row, in_h)) {
          int im_col = -pad_w + k_w * dilation;
          for (int w = 0; w < out_w; ++w, ++col_data, im_col += stride_w) {
            if (check_border(im_col, in_w)) {
              in_data[im_row * in_w + im_col] += *(col_data);
            }
          }
        } else {
          col_data += out_w;
        }
      }
    }
  }
}

template void Col2Im(const float *, const VecInt &, int, int, int, int, int,
                     int, int, int, const VecInt &, float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
