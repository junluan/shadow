#include "shadow/layers/conv_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

inline int convolutional_out_size(int s, int size, int pad, int stride) {
  return (s + 2 * pad - size) / stride + 1;
}

void ConvLayer::Setup(VecBlob *blobs) {
  Blob *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob *top = new Blob(layer_param_.top(0));

  if (!(bottom->shape(1) && bottom->shape(2) && bottom->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  num_output_ = layer_param_.convolution_param().num_output();
  kernel_size_ = layer_param_.convolution_param().kernel_size();
  stride_ = layer_param_.convolution_param().stride();
  pad_ = layer_param_.convolution_param().pad();
  activate_ = layer_param_.convolution_param().activate();

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_c = num_output_;
  int out_h = convolutional_out_size(in_h, kernel_size_, pad_, stride_);
  int out_w = convolutional_out_size(in_w, kernel_size_, pad_, stride_);

  *top->mutable_shape() = bottom->shape();
  top->set_shape(1, out_c);
  top->set_shape(2, out_h);
  top->set_shape(3, out_w);

  top->allocate_data(top->count());

  bottom_.push_back(bottom);
  top_.push_back(top);

  blobs->push_back(top);

  out_map_size_ = out_h * out_w;
  kernel_num_ = kernel_size_ * kernel_size_ * in_c;

  filters_ = new Blob(kernel_num_ * out_c);
  biases_ = new Blob(out_c);
  col_image_ = new Blob(out_map_size_ * kernel_num_);

#if defined(VERBOSE)
  std::cout << "Convolution Layer: " << format_vector(bottom->shape(), " x ")
            << " input -> " << out_c << "_" << kernel_size_ << "x"
            << kernel_size_ << "_s" << stride_ << "_p" << pad_ << " filters -> "
            << format_vector(top->shape(), " x ") << " output" << std::endl;
#endif
}

void ConvLayer::Forward() {
  const Blob *bottom = bottom_.at(0);
  Blob *top = top_.at(0);

  int batch = bottom->shape(0);
  int out_c = top->shape(1);
  BType *out_data = top->mutable_data();
  for (int b = 0; b < batch; ++b) {
    Blas::SetArrayRepeat(out_map_size_, biases_->data(), out_c, out_data,
                         b * top->num());
  }
  for (int b = 0; b < batch; ++b) {
    Image::Im2Col(bottom->shape(), bottom->data(), b * bottom->num(),
                  kernel_size_, stride_, pad_, top->shape(),
                  col_image_->mutable_data());
    Blas::BlasSGemm(0, 0, out_c, out_map_size_, kernel_num_, 1,
                    filters_->data(), kernel_num_, col_image_->data(),
                    out_map_size_, 1, out_data, b * top->num(), out_map_size_);
  }
  Activations::ActivateArray(top->count(), activate_, out_data);
}

void ConvLayer::Release() {
  bottom_.clear();
  top_.clear();

  filters_->clear();
  biases_->clear();
  col_image_->clear();

  // std::cout << "Free ConvLayer!" << std::endl;
}
