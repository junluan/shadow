#include "shadow/layers/convolution_layer.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

#if defined(USE_NNPACK)
#include "nnpack.h"
#endif

inline int convolution_out_size(int dim, int kernel_size, int stride, int pad,
                                int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return (dim + 2 * pad - kernel_extent) / stride + 1;
}

void ConvolutionLayer::Setup(VecBlob *blobs) {
  Layer::Setup(blobs);

  const auto &conv_param = layer_param_.convolution_param();

  num_output_ = conv_param.num_output();
  CHECK(conv_param.has_kernel_size());
  kernel_size_ = conv_param.kernel_size();
  stride_ = conv_param.stride();
  pad_ = conv_param.pad();
  dilation_ = conv_param.dilation();
  bias_term_ = conv_param.bias_term();
}

void ConvolutionLayer::Reshape() {
  int in_c = bottoms_[0]->shape(1), in_h = bottoms_[0]->shape(2),
      in_w = bottoms_[0]->shape(3);

  VecInt top_shape = bottoms_[0]->shape();
  top_shape[1] = num_output_;
  top_shape[2] =
      convolution_out_size(in_h, kernel_size_, stride_, pad_, dilation_);
  top_shape[3] =
      convolution_out_size(in_w, kernel_size_, stride_, pad_, dilation_);
  tops_[0]->reshape(top_shape);

  out_spatial_dim_ = tops_[0]->count(2);
  kernel_dim_ = kernel_size_ * kernel_size_ * in_c;

  biases_multiplier_.reshape(out_spatial_dim_);
  Blas::Set(out_spatial_dim_, 1, biases_multiplier_.mutable_data(), 0);
  col_image_.reshape(kernel_dim_, out_spatial_dim_);

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << num_output_ << "_" << kernel_size_ << "x"
             << kernel_size_ << "_s" << stride_ << "_p" << pad_ << " -> "
             << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ConvolutionLayer::Forward() {
  int batch = bottoms_[0]->shape(0);
  int top_num = tops_[0]->num(), bottom_num = bottoms_[0]->num();

#if defined(USE_NNPACK)
  if (batch == 1 && dilation_ == 1) {
    int in_c = bottoms_[0]->shape(1), out_c = tops_[0]->shape(1);

    nnp_size input_size;
    input_size.height = static_cast<size_t>(bottoms_[0]->shape(2));
    input_size.width = static_cast<size_t>(bottoms_[0]->shape(3));

    nnp_padding pad;
    pad.top = pad.bottom = pad.left = pad.right = static_cast<size_t>(pad_);

    nnp_size kernel_size;
    kernel_size.height = kernel_size.width = static_cast<size_t>(kernel_size_);

    nnp_size stride;
    stride.height = stride.width = static_cast<size_t>(stride_);

    auto algorithm = nnp_convolution_algorithm_auto;
    auto transform = nnp_convolution_transform_strategy_tuple_based;

    auto status = nnp_convolution_inference(
        algorithm, transform, in_c, out_c, input_size, pad, kernel_size, stride,
        bottoms_[0]->data(), blobs_[0]->data(), blobs_[1]->data(),
        tops_[0]->mutable_data(), nullptr, nullptr);
    CHECK_EQ(nnp_status_success, status);
    return;
  }
#endif

  for (int b = 0; b < batch; ++b) {
    Image::Im2Col(bottoms_[0]->data(), bottoms_[0]->shape(), b * bottom_num,
                  kernel_size_, stride_, pad_, dilation_, tops_[0]->shape(),
                  col_image_.mutable_data());
    Blas::BlasSgemm(0, 0, num_output_, out_spatial_dim_, kernel_dim_, 1,
                    blobs_[0]->data(), 0, col_image_.data(), 0, 0,
                    tops_[0]->mutable_data(), b * top_num);
    if (bias_term_) {
      Blas::BlasSgemm(0, 0, num_output_, out_spatial_dim_, 1, 1,
                      blobs_[1]->data(), 0, biases_multiplier_.data(), 0, 1,
                      tops_[0]->mutable_data(), b * top_num);
    }
  }
}

void ConvolutionLayer::Release() {
  biases_multiplier_.clear();
  col_image_.clear();

  // DLOG(INFO) << "Free ConvolutionLayer!";
}
