#include "shadow/layers/conv_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

inline int convolutional_out_size(int s, int size, int pad, int stride) {
  return (s + 2 * pad - size) / stride + 1;
}

void ConvLayer::Reshape() {
  num_output_ = layer_param_.convolution_param().num_output();
  kernel_size_ = layer_param_.convolution_param().kernel_size();
  stride_ = layer_param_.convolution_param().stride();
  pad_ = layer_param_.convolution_param().pad();
  activate_ = layer_param_.convolution_param().activate();

  int in_c = bottom_[0]->shape(1), in_h = bottom_[0]->shape(2),
      in_w = bottom_[0]->shape(3);
  int out_c = num_output_;
  int out_h = convolutional_out_size(in_h, kernel_size_, pad_, stride_);
  int out_w = convolutional_out_size(in_w, kernel_size_, pad_, stride_);

  *top_[0]->mutable_shape() = bottom_[0]->shape();
  top_[0]->set_shape(1, out_c);
  top_[0]->set_shape(2, out_h);
  top_[0]->set_shape(3, out_w);
  top_[0]->allocate_data(top_[0]->count());

  out_map_size_ = out_h * out_w;
  kernel_num_ = kernel_size_ * kernel_size_ * in_c;

  filters_ = new Blob<float>(kernel_num_ * out_c, layer_name_ + " filters");
  biases_ = new Blob<float>(out_c, layer_name_ + " biases");
  col_image_ =
      new Blob<float>(out_map_size_ * kernel_num_, layer_name_ + " col_image");

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << out_c << "_" << kernel_size_ << "x" << kernel_size_ << "_s" << stride_
      << "_p" << pad_ << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ConvLayer::Forward() {
  int batch = bottom_[0]->shape(0);
  int out_c = top_[0]->shape(1);
  int top_num = top_[0]->num(), bottom_num = bottom_[0]->num();
  for (int b = 0; b < batch; ++b) {
    Blas::SetArrayRepeat(out_map_size_, biases_->data(), out_c,
                         top_[0]->mutable_data(), b * top_num);
  }
  for (int b = 0; b < batch; ++b) {
    Image::Im2Col(bottom_[0]->shape(), bottom_[0]->data(), b * bottom_num,
                  kernel_size_, stride_, pad_, top_[0]->shape(),
                  col_image_->mutable_data());
    Blas::BlasSGemm(0, 0, out_c, out_map_size_, kernel_num_, 1,
                    filters_->data(), kernel_num_, col_image_->data(),
                    out_map_size_, 1, top_[0]->mutable_data(), b * top_num,
                    out_map_size_);
  }
  Activations::ActivateArray(top_[0]->count(), activate_,
                             top_[0]->mutable_data());
}

void ConvLayer::Release() {
  bottom_.clear();
  top_.clear();

  filters_->clear();
  biases_->clear();
  col_image_->clear();

  // std::cout << "Free ConvLayer!" << std::endl;
}
