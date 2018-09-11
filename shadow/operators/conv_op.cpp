#include "conv_op.hpp"

#include "activate_op.hpp"

namespace Shadow {

void ConvOp::Forward() {
  if (bias_term_) {
    CHECK_EQ(bottoms_size(), 3);
  } else {
    CHECK_EQ(bottoms_size(), 2);
  }

  const auto *bottom = bottoms<float>(0);
  const auto *weight = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int batch = bottom->shape(0), in_c = bottom->shape(1),
      in_h = bottom->shape(2), in_w = bottom->shape(3);

  CHECK_EQ(in_c % group_, 0);

  auto top_shape = bottom->shape();
  top_shape[1] = num_output_;
  top_shape[2] = conv_out_size(in_h, kernel_size_, stride_, pad_, dilation_);
  top_shape[3] = conv_out_size(in_w, kernel_size_, stride_, pad_, dilation_);
  top->reshape(top_shape);

  out_spatial_dim_ = top->count(2);
  kernel_dim_ = kernel_size_ * kernel_size_ * in_c / group_;

  weight_offset_ = num_output_ * kernel_dim_ / group_;
  col_offset_ = kernel_dim_ * out_spatial_dim_;
  output_offset_ = num_output_ * out_spatial_dim_ / group_;

#if defined(USE_NNPACK)
  use_nnpack_ = batch == 1 && group_ == 1 && dilation_ == 1 && bias_term_;
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

    int out_c = top->shape(1);
    auto status = nnp_convolution_inference(
        nnp_algorithm_, nnp_transform_, in_c, out_c, nnp_input_size_, nnp_pad_,
        nnp_kernel_size_, nnp_stride_, bottom->data(), weight->data(),
        bottoms<float>(2)->data(), top->mutable_data(), nullptr, nullptr,
        nnp_activation_, nullptr, Kernel::nnp_pthreadpool_, nullptr);
    CHECK_EQ(nnp_status_success, status);
    return;
  }
#endif

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    cudnn::setConvolution2dDesc<float>(&conv_desc_, pad_, pad_, stride_,
                                       stride_, dilation_, dilation_, group_);
    cudnn::setTensor4dDesc<float>(&bottom_desc_, batch, in_c, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&top_desc_, batch, num_output_, top_shape[2],
                                  top_shape[3]);
    cudnn::setFilter4dDesc<float>(&filter_desc_, num_output_, in_c / group_,
                                  kernel_size_, kernel_size_);
    if (bias_term_) {
      cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output_, 1, 1);
    }

    size_t workspace_limit_bytes = group_ == 1 ? 64 * 1024 * 1024 : 0;

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        Kernel::cudnn_handle_, bottom_desc_, filter_desc_, conv_desc_,
        top_desc_, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &fwd_algo_));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        Kernel::cudnn_handle_, bottom_desc_, filter_desc_, conv_desc_,
        top_desc_, fwd_algo_, &workspace_fwd_size_));

    if (workspace_fwd_size_ > 0) {
      op_ws_->GrowTempBuffer(static_cast<int>(workspace_fwd_size_) *
                             sizeof(unsigned char));
      workspace_ = op_ws_->CreateTempBlob<unsigned char>(
          {static_cast<int>(workspace_fwd_size_)}, op_name_ + "_workspace");
    }

    auto *workspace_ptr =
        workspace_fwd_size_ > 0 ? workspace_->mutable_data() : nullptr;
    CUDNN_CHECK(cudnnConvolutionForward(
        Kernel::cudnn_handle_, cudnn::dataType<float>::one, bottom_desc_,
        bottom->data(), filter_desc_, weight->data(), conv_desc_, fwd_algo_,
        workspace_ptr, workspace_fwd_size_, cudnn::dataType<float>::zero,
        top_desc_, top->mutable_data()));
    if (bias_term_) {
      CUDNN_CHECK(cudnnAddTensor(
          Kernel::cudnn_handle_, cudnn::dataType<float>::one, bias_desc_,
          bottoms<float>(2)->data(), cudnn::dataType<float>::one, top_desc_,
          top->mutable_data()));
    }
    if (activate_type_ == 1) {
      Vision::Activate(top->mutable_data(), top->count(), activate_type_);
    }
    return;
  }
#endif

  use_depthwise_ = dilation_ == 1 && group_ == in_c && group_ == num_output_;
  if (use_depthwise_) {
    if (bias_term_) {
      Vision::Depthwise(bottom->data(), bottom->shape(), weight->data(),
                        bottoms<float>(2)->data(), kernel_size_, stride_, pad_,
                        bias_term_, top->shape(), top->mutable_data());
    } else {
      Vision::Depthwise(bottom->data(), bottom->shape(), weight->data(),
                        static_cast<decltype(weight->data())>(nullptr),
                        kernel_size_, stride_, pad_, bias_term_, top->shape(),
                        top->mutable_data());
    }
  } else {
    int temp_count = kernel_dim_ * group_ * out_spatial_dim_;
    if (bias_term_) {
      temp_count += out_spatial_dim_;
    }
    op_ws_->GrowTempBuffer(temp_count * sizeof(float));
    col_image_ = op_ws_->CreateTempBlob<float>(
        {kernel_dim_ * group_, out_spatial_dim_}, op_name_ + "_col_image");
    if (bias_term_) {
      biases_multiplier_ = op_ws_->CreateTempBlob<float>(
          {out_spatial_dim_}, op_name_ + "_biases_multiplier");
      Blas::Set(out_spatial_dim_, 1, biases_multiplier_->mutable_data(), 0);
    }
    int top_num = top->num(), bottom_num = bottom->num();
    for (int b = 0; b < batch; ++b) {
      Vision::Im2Col(bottom->data(), bottom->shape(), b * bottom_num,
                     kernel_size_, stride_, pad_, dilation_, 0, top->shape(),
                     col_image_->mutable_data());
      for (int g = 0; g < group_; ++g) {
        Blas::BlasSgemm(0, 0, num_output_ / group_, out_spatial_dim_,
                        kernel_dim_, 1, weight->data(), weight_offset_ * g,
                        col_image_->data(), col_offset_ * g, 0,
                        top->mutable_data(), b * top_num + output_offset_ * g);
      }
      if (bias_term_) {
        Blas::BlasSgemm(0, 0, num_output_, out_spatial_dim_, 1, 1,
                        bottoms<float>(2)->data(), 0,
                        biases_multiplier_->data(), 0, 1, top->mutable_data(),
                        b * top_num);
      }
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

template <typename T>
void Depthwise(const T *in_data, const VecInt &in_shape, const T *weight_data,
               const T *bias_data, int kernel_size, int stride, int pad,
               int bias_term, const VecInt &out_shape, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      const T *in_offset_data = in_data + (b * in_c + c) * in_h * in_w;
      const T *weight_offset_data = weight_data + c * kernel_size * kernel_size;
      T *out_offset_data = out_data + (b * in_c + c) * out_h * out_w;
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int hstart = h * stride - pad, wstart = w * stride - pad;
          int hend = std::min(hstart + kernel_size, in_h + pad);
          int wend = std::min(wstart + kernel_size, in_w + pad);
          hstart = std::max(hstart, 0), wstart = std::max(wstart, 0);
          hend = std::min(hend, in_h), wend = std::min(wend, in_w);
          int khstart = hend < kernel_size ? (kernel_size - hend) : 0;
          int kwstart = wend < kernel_size ? (kernel_size - wend) : 0;
          auto sum_val = T(0);
          for (int kh = hstart; kh < hend; ++kh) {
            for (int kw = wstart; kw < wend; ++kw) {
              sum_val +=
                  in_offset_data[kh * in_w + kw] *
                  weight_offset_data[(khstart + kh - hstart) * kernel_size +
                                     kwstart + kw - wstart];
            }
          }
          if (bias_term) {
            sum_val += bias_data[c];
          }
          out_offset_data[h * out_w + w] = sum_val;
        }
      }
    }
  }
}

template void Depthwise(const float *in_data, const VecInt &in_shape,
                        const float *weight_data, const float *bias_data,
                        int kernel_size, int stride, int pad, int bias_term,
                        const VecInt &out_shape, float *out_data);

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
