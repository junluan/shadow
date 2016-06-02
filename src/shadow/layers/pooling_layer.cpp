#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

PoolingLayer::PoolingLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob_ = new Blob<BType>();
  out_blob_ = new Blob<BType>();
}
PoolingLayer::~PoolingLayer() { ReleaseLayer(); }

void PoolingLayer::MakeLayer(Blob<BType> *blob) {
  if (!(blob->shape(1) && blob->shape(2) && blob->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();

  int batch = blob->shape(0);
  int in_c = blob->shape(1), in_h = blob->shape(2), in_w = blob->shape(3);
  int out_c = in_c;
  int out_h = (in_h - kernel_size_) / stride_ + 1;
  int out_w = (in_w - kernel_size_) / stride_ + 1;

  int in_num = in_c * in_h * in_w;
  int out_num = out_c * out_h * out_w;

  *in_blob_->mutable_shape() = blob->shape();
  blob->set_shape(2, out_h);
  blob->set_shape(3, out_w);
  *out_blob_->mutable_shape() = blob->shape();

  in_blob_->set_num(in_num);
  out_blob_->set_num(out_num);
  in_blob_->set_count(batch * in_num);
  out_blob_->set_count(batch * out_num);

  out_blob_->allocate_data(out_blob_->count());

#if defined(VERBOSE)
  printf(
      "Maxpool Layer: %d x %d x %d input -> %dx%d_s%d -> %d x %d x %d output\n",
      in_c, in_h, in_w, kernel_size_, kernel_size_, stride_, out_c, out_h,
      out_w);
#endif
}

void PoolingLayer::ForwardLayer() {
  int batch = in_blob_->shape(0), in_c = in_blob_->shape(1);
  int in_h = in_blob_->shape(2), in_w = in_blob_->shape(3);
  int out_h = out_blob_->shape(2), out_w = out_blob_->shape(3);
  Image::Pooling(in_blob_->data(), batch, in_c, in_h, in_w, kernel_size_,
                 stride_, out_h, out_w, pool_type_, out_blob_->mutable_data());
}

void PoolingLayer::ReleaseLayer() {
  out_blob_->clear();
  // std::cout << "Free PoolingLayer!" << std::endl;
}
