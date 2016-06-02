#include "shadow/layers/conv_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

inline int convolutional_out_size(int s, int size, int pad, int stride) {
  return (s + 2 * pad - size) / stride + 1;
}

ConvLayer::ConvLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob_ = new Blob<BType>();
  out_blob_ = new Blob<BType>();
  filters_ = new Blob<BType>();
  biases_ = new Blob<BType>();
  col_image_ = new Blob<BType>();
}
ConvLayer::~ConvLayer() { ReleaseLayer(); }

void ConvLayer::MakeLayer(Blob<BType> *blob) {
  if (!(blob->shape(1) && blob->shape(2) && blob->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  num_output_ = layer_param_.convolution_param().num_output();
  kernel_size_ = layer_param_.convolution_param().kernel_size();
  stride_ = layer_param_.convolution_param().stride();
  pad_ = layer_param_.convolution_param().pad();
  activate_ = layer_param_.convolution_param().activate();

  int batch = blob->shape(0);
  int in_c = blob->shape(1), in_h = blob->shape(2), in_w = blob->shape(3);
  int out_c = num_output_;
  int out_h = convolutional_out_size(in_h, kernel_size_, pad_, stride_);
  int out_w = convolutional_out_size(in_w, kernel_size_, pad_, stride_);

  int in_num = in_c * in_h * in_w;
  int out_num = out_c * out_h * out_w;

  *in_blob_->mutable_shape() = blob->shape();
  blob->set_shape(1, out_c);
  blob->set_shape(2, out_h);
  blob->set_shape(3, out_w);
  *out_blob_->mutable_shape() = blob->shape();

  in_blob_->set_num(in_num);
  out_blob_->set_num(out_num);
  in_blob_->set_count(batch * in_num);
  out_blob_->set_count(batch * out_num);

  out_map_size_ = out_h * out_w;
  kernel_num_ = kernel_size_ * kernel_size_ * in_c;

  out_blob_->allocate_data(out_blob_->count());

  filters_->allocate_data(kernel_num_ * out_c);
  biases_->allocate_data(out_c);
  col_image_->allocate_data(out_map_size_ * kernel_num_);

#if defined(VERBOSE)
  printf(
      "Convolutional Layer: %d x %d x %d input -> %d_%dx%d_s%d_p%d filters -> "
      "%d x %d x %d output\n",
      in_c, in_h, in_w, out_c, kernel_size_, kernel_size_, stride_, pad_, out_c,
      out_h, out_w);
#endif
}

void ConvLayer::ForwardLayer() {
  int batch = in_blob_->shape(0), in_c = in_blob_->shape(1);
  int in_h = in_blob_->shape(2), in_w = in_blob_->shape(3);
  int out_c = out_blob_->shape(1);
  int out_h = out_blob_->shape(2), out_w = out_blob_->shape(3);
  BType *out_data = out_blob_->mutable_data();
  for (int b = 0; b < batch; ++b) {
    Blas::SetArrayRepeat(out_map_size_, biases_->data(), out_c,
                         out_data + b * out_blob_->num());
  }
  for (int b = 0; b < batch; ++b) {
    Image::Im2Col(in_blob_->data(), b * in_blob_->num(), in_c, in_h, in_w,
                  kernel_size_, stride_, pad_, out_h, out_w,
                  col_image_->mutable_data());
    Blas::BlasSGemm(0, 0, out_c, out_map_size_, kernel_num_, 1,
                    filters_->data(), kernel_num_, col_image_->data(),
                    out_map_size_, 1, out_data, b * out_blob_->num(),
                    out_map_size_);
  }
  Activations::ActivateArray(out_blob_->count(), activate_, out_data);
}

void ConvLayer::ReleaseLayer() {
  out_blob_->clear();

  filters_->clear();
  biases_->clear();
  col_image_->clear();
  // std::cout << "Free ConvLayer!" << std::endl;
}
