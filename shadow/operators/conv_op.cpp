#include "conv_op.hpp"

#include "activate_op.hpp"

namespace Shadow {

void ConvOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int batch = bottom->shape(0), in_c = bottom->shape(1),
      in_h = bottom->shape(2), in_w = bottom->shape(3);

  VecInt top_shape = bottom->shape();
  top_shape[1] = num_output_;
  top_shape[2] = conv_out_size(in_h, kernel_size_, stride_, pad_, dilation_);
  top_shape[3] = conv_out_size(in_w, kernel_size_, stride_, pad_, dilation_);
  top->reshape(top_shape);

  out_spatial_dim_ = top->count(2);
  kernel_dim_ = kernel_size_ * kernel_size_ * in_c / group_;

  weight_offset_ = num_output_ * kernel_dim_ / group_;
  col_offset_ = kernel_dim_ * out_spatial_dim_;
  output_offset_ = num_output_ * out_spatial_dim_ / group_;

  if (!use_cudnn_ && !use_nnpack_) {
    if (bias_term_) {
      biases_multiplier_ =
          op_ws_->CreateBlob<float>(op_name_ + "_biases_multiplier");
      biases_multiplier_->reshape({out_spatial_dim_});
      Blas::Set(out_spatial_dim_, 1, biases_multiplier_->mutable_data(), 0);
    }
    col_image_ = op_ws_->CreateBlob<float>(op_name_ + "_col_image");
    col_image_->reshape({kernel_dim_ * group_, out_spatial_dim_});
  }

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    cudnn::setTensor4dDesc<float>(&bottom_desc_, batch, in_c, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&top_desc_, batch, num_output_, top_shape[2],
                                  top_shape[3]);
    cudnn::setConvolutionDesc<float>(&conv_desc_, bottom_desc_, filter_desc_,
                                     pad_, pad_, stride_, stride_);

    size_t workspace_limit_bytes = 64 * 1024 * 1024;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        Kernel::cudnn_handle_, bottom_desc_, filter_desc_, conv_desc_,
        top_desc_, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &fwd_algo_));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        Kernel::cudnn_handle_, bottom_desc_, filter_desc_, conv_desc_,
        top_desc_, fwd_algo_, &(workspace_fwd_size_)));

    if (workspace_fwd_size_ > 0) {
      cudaFree(workspace_);
      if (cudaMalloc(&workspace_, workspace_fwd_size_) != cudaSuccess) {
        fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        workspace_fwd_size_ = 0;
        workspace_ = nullptr;
      }
    }
  }
#endif

#if defined(USE_NNPACK)
  if (use_nnpack_) {
    nnp_algorithm_ = nnp_convolution_algorithm_auto;
    nnp_transform_ = nnp_convolution_transform_strategy_compute;
    nnp_activation_ =
        activate_type_ == 1 ? nnp_activation_relu : nnp_activation_identity;
    nnp_input_size_.height = static_cast<size_t>(in_h);
    nnp_input_size_.width = static_cast<size_t>(in_w);
    nnp_pad_.top = nnp_pad_.bottom = nnp_pad_.left = nnp_pad_.right =
        static_cast<size_t>(pad_);
    nnp_kernel_size_.height = nnp_kernel_size_.width =
        static_cast<size_t>(kernel_size_);
    nnp_stride_.height = nnp_stride_.width = static_cast<size_t>(stride_);
  }
#endif

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << num_output_ << "_" << kernel_size_ << "x" << kernel_size_
             << "_s" << stride_ << "_p" << pad_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void ConvOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0);
  int top_num = top->num(), bottom_num = bottom->num();

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    CUDNN_CHECK(cudnnConvolutionForward(
        Kernel::cudnn_handle_, cudnn::dataType<float>::one, bottom_desc_,
        bottom->data(), filter_desc_, blobs<float>(0)->data(), conv_desc_,
        fwd_algo_, workspace_, workspace_fwd_size_,
        cudnn::dataType<float>::zero, top_desc_, top->mutable_data()));
    if (bias_term_) {
      CUDNN_CHECK(cudnnAddTensor(
          Kernel::cudnn_handle_, cudnn::dataType<float>::one, bias_desc_,
          blobs<float>(1)->data(), cudnn::dataType<float>::one, top_desc_,
          top->mutable_data()));
    }
    if (activate_type_ == 1) {
      Vision::Activate(top->mutable_data(), top->count(), activate_type_);
    }
    return;
  }
#endif

#if defined(USE_NNPACK)
  if (use_nnpack_) {
    int in_c = bottom->shape(1), out_c = top->shape(1);
    auto status = nnp_convolution_inference(
        nnp_algorithm_, nnp_transform_, in_c, out_c, nnp_input_size_, nnp_pad_,
        nnp_kernel_size_, nnp_stride_, bottom->data(), blobs<float>(0)->data(),
        blobs<float>(1)->data(), top->mutable_data(), nullptr, nullptr,
        nnp_activation_, nullptr, Kernel::nnp_pthreadpool_, nullptr);
    CHECK_EQ(nnp_status_success, status);
    return;
  }
#endif

  for (int b = 0; b < batch; ++b) {
    Vision::Im2Col(bottom->data(), bottom->shape(), b * bottom_num,
                   kernel_size_, stride_, pad_, dilation_, 0, top->shape(),
                   col_image_->mutable_data());
    for (int g = 0; g < group_; ++g) {
      Blas::BlasSgemm(0, 0, num_output_ / group_, out_spatial_dim_, kernel_dim_,
                      1, blobs<float>(0)->data(), weight_offset_ * g,
                      col_image_->data(), col_offset_ * g, 0,
                      top->mutable_data(), b * top_num + output_offset_ * g);
    }
    if (bias_term_) {
      Blas::BlasSgemm(0, 0, num_output_, out_spatial_dim_, 1, 1,
                      blobs<float>(1)->data(), 0, biases_multiplier_->data(), 0,
                      1, top->mutable_data(), b * top_num);
    }
  }
  if (activate_type_ == 1) {
    Vision::Activate(top->mutable_data(), top->count(), activate_type_);
  }
}

REGISTER_OPERATOR(Conv, ConvOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
// check for 0 <= a < b
inline bool check_border(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation, int zero_point,
            const VecInt &out_shape, T *col_data) {
  in_data += offset;
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int spatial_dim = in_h * in_w;
  for (int k_c = 0; k_c < in_c; ++k_c, in_data += spatial_dim) {
    for (int k_s = 0; k_s < kernel_size * kernel_size; ++k_s) {
      int k_h = k_s / kernel_size;
      int k_w = k_s % kernel_size;
      int im_row = -pad + k_h * dilation;
      for (int h = 0; h < out_h; ++h, im_row += stride) {
        if (check_border(im_row, in_h)) {
          int im_col = -pad + k_w * dilation;
          for (int w = 0; w < out_w; ++w, im_col += stride) {
            if (check_border(im_col, in_w)) {
              *(col_data++) = in_data[im_row * in_w + im_col];
            } else {
              *(col_data++) = static_cast<T>(zero_point);
            }
          }
        } else {
          for (int w = 0; w < out_w; ++w) {
            *(col_data++) = static_cast<T>(zero_point);
          }
        }
      }
    }
  }
}

template void Im2Col(const float *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     int zero_point, const VecInt &out_shape, float *col_data);
template void Im2Col(const unsigned char *in_data, const VecInt &in_shape,
                     int offset, int kernel_size, int stride, int pad,
                     int dilation, int zero_point, const VecInt &out_shape,
                     unsigned char *col_data);

#elif defined(USE_CL)
template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation, int zero_point,
            const VecInt &out_shape, T *col_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Im2Col"];
  kernel->SetArguments(*in_data, offset, count, in_c, in_h, in_w, kernel_size,
                       stride, pad, dilation, zero_point, out_h, out_w,
                       *col_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Im2Col(const BufferF *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     int zero_point, const VecInt &out_shape,
                     BufferF *col_data);
#endif

}  // namespace Vision

}  // namespace Shadow
