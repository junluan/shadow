#include "conv_op.hpp"
#include "core/blas.hpp"
#include "core/image.hpp"

namespace Shadow {

inline int conv_out_size(int dim, int kernel_size, int stride, int pad,
                         int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return (dim + 2 * pad - kernel_extent) / stride + 1;
}

void ConvOp::Setup() {
  num_output_ = get_single_argument<int>("num_output", 0);
  CHECK(has_argument("kernel_size"));
  kernel_size_ = get_single_argument<int>("kernel_size", 0);
  stride_ = get_single_argument<int>("stride", 1);
  pad_ = get_single_argument<int>("pad", 0);
  dilation_ = get_single_argument<int>("dilation", 1);
  group_ = get_single_argument<int>("group", 1);
  CHECK_EQ(bottoms<float>(0)->shape(1) % group_, 0);
  CHECK_EQ(num_output_ % group_, 0);
  bias_term_ = get_single_argument<bool>("bias_term", true);

  if (bias_term_) {
    CHECK_EQ(blobs_size(), 2);
  } else {
    CHECK_EQ(blobs_size(), 1);
  }

#if defined(USE_CUDNN)
  use_cudnn_ = dilation_ == 1 && group_ == 1;
  if (use_cudnn_) {
    cudnn::createConvolutionDesc<float>(&conv_desc_);
    cudnn::createTensor4dDesc<float>(&bottom_desc_);
    cudnn::createTensor4dDesc<float>(&top_desc_);
    cudnn::createFilterDesc<float>(&filter_desc_, num_output_,
                                   bottoms<float>(0)->shape(1), kernel_size_,
                                   kernel_size_);
    if (bias_term_) {
      cudnn::createTensor4dDesc<float>(&bias_desc_);
      cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output_, 1, 1);
    }
  }
#endif

#if defined(USE_NNPACK)
  use_nnpack_ = bottoms<float>(0)->shape(0) == 1 && group_ == 1 &&
                dilation_ == 1 && bias_term_;
#endif
}

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
      biases_multiplier_ = op_ws_->CreateBlob<float>(
          {out_spatial_dim_}, op_name_ + "_biases_multiplier");
      Blas::Set(out_spatial_dim_, 1, biases_multiplier_->mutable_data(), 0);
    }
    col_image_ = op_ws_->CreateBlob<float>(
        {kernel_dim_ * group_, out_spatial_dim_}, op_name_ + "_col_image");
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
        nnp_activation_identity, nullptr, Kernel::nnp_pthreadpool_, nullptr);
    CHECK_EQ(nnp_status_success, status);
    return;
  }
#endif

  for (int b = 0; b < batch; ++b) {
    Image::Im2Col(bottom->data(), bottom->shape(), b * bottom_num, kernel_size_,
                  stride_, pad_, dilation_, 0, top->shape(),
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
}

void ConvOp::Release() {
#if defined(USE_CUDNN)
  if (conv_desc_ != nullptr) {
    cudnnDestroyConvolutionDescriptor(conv_desc_);
    conv_desc_ = nullptr;
  }
  if (bottom_desc_ != nullptr) {
    cudnnDestroyTensorDescriptor(bottom_desc_);
    bottom_desc_ = nullptr;
  }
  if (top_desc_ != nullptr) {
    cudnnDestroyTensorDescriptor(top_desc_);
    top_desc_ = nullptr;
  }
  if (filter_desc_ != nullptr) {
    cudnnDestroyFilterDescriptor(filter_desc_);
    filter_desc_ = nullptr;
  }
  if (bias_desc_ != nullptr) {
    cudnnDestroyTensorDescriptor(bias_desc_);
    bias_desc_ = nullptr;
  }

  if (workspace_ != nullptr) {
    cudaFree(workspace_);
    workspace_ = nullptr;
  }
#endif

  // DLOG(INFO) << "Free ConvOp!";
}

REGISTER_OPERATOR(Conv, ConvOp);

}  // namespace Shadow
