#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

void PoolingLayer::Setup(VecBlob *blobs) {
  Blob *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob *top = new Blob(layer_param_.top(0));

  if (!(bottom->shape(1) && bottom->shape(2) && bottom->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_c = in_c;
  int out_h = (in_h - kernel_size_) / stride_ + 1;
  int out_w = (in_w - kernel_size_) / stride_ + 1;

  *top->mutable_shape() = bottom->shape();
  top->set_shape(2, out_h);
  top->set_shape(3, out_w);

  top->allocate_data(top->count());

  bottom_.push_back(bottom);
  top_.push_back(top);

  blobs->push_back(top);

#if defined(VERBOSE)
  printf(
      "Maxpool Layer: %d x %d x %d input -> %dx%d_s%d -> %d x %d x %d output\n",
      in_c, in_h, in_w, kernel_size_, kernel_size_, stride_, out_c, out_h,
      out_w);
#endif
}

void PoolingLayer::Forward() {
  Blob *bottom = bottom_.at(0);
  Blob *top = top_.at(0);

  int batch = bottom->shape(0), in_c = bottom->shape(1);
  int in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_h = top->shape(2), out_w = top->shape(3);
  Image::Pooling(bottom->data(), batch, in_c, in_h, in_w, kernel_size_, stride_,
                 out_h, out_w, pool_type_, top->mutable_data());
}

void PoolingLayer::Release() {
  bottom_.clear();
  top_.clear();

  // std::cout << "Free PoolingLayer!" << std::endl;
}
