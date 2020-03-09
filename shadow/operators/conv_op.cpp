#include "conv_op.hpp"

#include "activate_op.hpp"

namespace Shadow {

void ConvOp::Forward() {
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
      conv_out_size(in_h, kernel_size_h_, stride_h_, pad_h_, dilation_);
  top_shape[3] =
      conv_out_size(in_w, kernel_size_w_, stride_w_, pad_w_, dilation_);
  top->reshape(top_shape);

  out_spatial_dim_ = top->count(2);
  kernel_dim_ = kernel_size_h_ * kernel_size_w_ * in_c / group_;

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
    nnp_kernel_size_.height = static_cast<size_t>(kernel_size_h_);
    nnp_kernel_size_.width = static_cast<size_t>(kernel_size_w_);
    nnp_stride_.height = static_cast<size_t>(stride_h_);
    nnp_stride_.width = static_cast<size_t>(stride_w_);
    nnp_pad_.top = nnp_pad_.bottom = static_cast<size_t>(pad_h_);
    nnp_pad_.left = nnp_pad_.right = static_cast<size_t>(pad_w_);

    int out_c = top->shape(1);
    auto status = nnp_convolution_inference(
        nnp_algorithm_, nnp_transform_, in_c, out_c, nnp_input_size_, nnp_pad_,
        nnp_kernel_size_, nnp_stride_, bottom->data<float>(),
        weight->data<float>(), bottoms(2)->data<float>(),
        top->mutable_data<float>(), nullptr, nullptr, nnp_activation_, nullptr,
        pthreadpool_t(ws_->Ctx()->nnpack_handle()), nullptr);
    CHECK_EQ(nnp_status_success, status);

    return;
  }
#endif

#if defined(USE_DNNL)
  if (!use_nnpack_) {
    const auto &src_desc = idnnl::create_memory_desc<float>(bottom->shape());
    const auto &dst_desc = idnnl::create_memory_desc<float>(top->shape());
    dnnl::memory::desc weight_desc;
    if (group_ == 1) {
      weight_desc = idnnl::create_memory_desc<float>(
          {num_output_, in_c, kernel_size_h_, kernel_size_w_},
          dnnl::memory::format_tag::oihw);
    } else {
      weight_desc = idnnl::create_memory_desc<float>(
          {group_, num_output_ / group_, in_c / group_, kernel_size_h_,
           kernel_size_w_},
          dnnl::memory::format_tag::goihw);
    }
    const auto &bias_desc = idnnl::create_memory_desc<float>(
        {num_output_}, bias_term_ ? dnnl::memory::format_tag::x
                                  : dnnl::memory::format_tag::undef);

    const auto &conv_desc = idnnl::create_convolution_desc(
        src_desc, weight_desc, bias_desc, dst_desc, pad_h_, pad_w_, stride_h_,
        stride_w_, dilation_, dilation_);

    idnnl::convolution_forward(ws_->Ctx()->dnnl_engine(),
                               ws_->Ctx()->dnnl_stream(), conv_desc,
                               bottom->data<float>(), weight->data<float>(),
                               bias_term_ ? bottoms(2)->data<float>() : nullptr,
                               top->mutable_data<float>(), activate_type_);

    return;
  }
#endif

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    cudnn::setConvolution2dDesc<float>(&conv_desc_, pad_h_, pad_w_, stride_h_,
                                       stride_w_, dilation_, dilation_, group_);
    cudnn::setTensor4dDesc<float>(&bottom_desc_, batch, in_c, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&top_desc_, batch, num_output_, top_shape[2],
                                  top_shape[3]);
    cudnn::setFilter4dDesc<float>(&filter_desc_, num_output_, in_c / group_,
                                  kernel_size_h_, kernel_size_w_);
    if (bias_term_) {
      cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output_, 1, 1);
    }
    if (activate_type_ == 1) {
      cudnn::setActivationDesc<float>(&activate_desc_, activate_type_, 0);
    }

    size_t workspace_limit_bytes = group_ == 1 ? 64 * 1024 * 1024 : 0;

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle_t(ws_->Ctx()->cudnn_handle()), bottom_desc_, filter_desc_,
        conv_desc_, top_desc_, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &fwd_algo_));

    size_t workspace_fwd_size = 0;

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle_t(ws_->Ctx()->cudnn_handle()), bottom_desc_, filter_desc_,
        conv_desc_, top_desc_, fwd_algo_, &workspace_fwd_size));

    std::shared_ptr<Blob> workspace = nullptr;
    const void *workspace_ptr = nullptr;
    if (workspace_fwd_size > 0) {
      ws_->GrowTempBuffer(workspace_fwd_size);
      workspace = ws_->CreateTempBlob({static_cast<int>(workspace_fwd_size)},
                                      DataType::kU8);
      workspace_ptr = workspace->data<unsigned char>();
    }

    CUDNN_CHECK(cudnnConvolutionForward(
        cudnnHandle_t(ws_->Ctx()->cudnn_handle()), cudnn::dataType<float>::one,
        bottom_desc_, bottom->data<float>(), filter_desc_,
        weight->data<float>(), conv_desc_, fwd_algo_,
        const_cast<void *>(workspace_ptr), workspace_fwd_size,
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

  use_depthwise_ = group_ == in_c && group_ == num_output_;
  if (use_depthwise_) {
    if (bias_term_) {
      Vision::Depthwise(bottom->data<float>(), bottom->shape(),
                        weight->data<float>(), bottoms(2)->data<float>(),
                        kernel_size_h_, kernel_size_w_, stride_h_, stride_w_,
                        pad_h_, pad_w_, dilation_, bias_term_, top->shape(),
                        top->mutable_data<float>(), ws_->Ctx());
    } else {
      Vision::Depthwise(
          bottom->data<float>(), bottom->shape(), weight->data<float>(),
          static_cast<decltype(weight->data<float>())>(nullptr), kernel_size_h_,
          kernel_size_w_, stride_h_, stride_w_, pad_h_, pad_w_, dilation_,
          bias_term_, top->shape(), top->mutable_data<float>(), ws_->Ctx());
    }
  } else {
    int temp_count = kernel_dim_ * group_ * out_spatial_dim_;
    if (bias_term_) {
      temp_count += out_spatial_dim_;
    }
    ws_->GrowTempBuffer(temp_count * sizeof(float));
    auto col_image = ws_->CreateTempBlob(
        {kernel_dim_ * group_, out_spatial_dim_}, DataType::kF32);
    std::shared_ptr<Blob> biases_multiplier = nullptr;
    if (bias_term_) {
      biases_multiplier =
          ws_->CreateTempBlob({out_spatial_dim_}, DataType::kF32);
      Blas::Set(out_spatial_dim_, 1, biases_multiplier->mutable_data<float>(),
                0, ws_->Ctx());
    }
    int top_num = top->num(), bottom_num = bottom->num();
    for (int b = 0; b < batch; ++b) {
      Vision::Im2Col(bottom->data<float>(), bottom->shape(), b * bottom_num,
                     kernel_size_h_, kernel_size_w_, stride_h_, stride_w_,
                     pad_h_, pad_w_, dilation_, 0, top->shape(),
                     col_image->mutable_data<float>(), ws_->Ctx());
      for (int g = 0; g < group_; ++g) {
        Blas::BlasSgemm(0, 0, num_output_ / group_, out_spatial_dim_,
                        kernel_dim_, 1, weight->data<float>(),
                        weight_offset_ * g, col_image->data<float>(),
                        col_offset_ * g, 0, top->mutable_data<float>(),
                        b * top_num + output_offset_ * g, ws_->Ctx());
      }
      if (bias_term_) {
        Blas::BlasSgemm(0, 0, num_output_, out_spatial_dim_, 1, 1,
                        bottoms(2)->data<float>(), 0,
                        biases_multiplier->data<float>(), 0, 1,
                        top->mutable_data<float>(), b * top_num, ws_->Ctx());
      }
    }
  }
  if (activate_type_ == 1) {
    Vision::Activate(top->data<float>(), top->mutable_data<float>(),
                     top->count(), activate_type_, 0, ws_->Ctx());
  }
}

REGISTER_OPERATOR(Conv, ConvOp);

namespace Vision {

#if !defined(USE_CUDA)
// check for 0 <= a < b
inline bool check_border(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
            int pad_h, int pad_w, int dilation, int zero_point,
            const VecInt &out_shape, T *col_data, Context *context) {
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
          for (int w = 0; w < out_w; ++w, im_col += stride_w) {
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

template void Im2Col(const float *, const VecInt &, int, int, int, int, int,
                     int, int, int, int, const VecInt &, float *, Context *);

template <typename T>
void Depthwise(const T *in_data, const VecInt &in_shape, const T *weight_data,
               const T *bias_data, int kernel_size_h, int kernel_size_w,
               int stride_h, int stride_w, int pad_h, int pad_w, int dilation,
               int bias_term, const VecInt &out_shape, T *out_data,
               Context *context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      const T *in_offset_data = in_data + (b * in_c + c) * in_h * in_w;
      T *out_offset_data = out_data + (b * in_c + c) * out_h * out_w;
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          const T *weight_offset_data =
              weight_data + c * kernel_size_h * kernel_size_w;
          auto sum_val = T(0);
          for (int kh = 0; kh < kernel_size_h; ++kh) {
            for (int kw = 0; kw < kernel_size_w; ++kw) {
              int h_in = h * stride_h - pad_h + kh * dilation;
              int w_in = w * stride_w - pad_w + kw * dilation;
              if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                sum_val +=
                    in_offset_data[h_in * in_w + w_in] * *weight_offset_data;
              }
              weight_offset_data++;
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

template void Depthwise(const float *, const VecInt &, const float *,
                        const float *, int, int, int, int, int, int, int, int,
                        const VecInt &, float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
