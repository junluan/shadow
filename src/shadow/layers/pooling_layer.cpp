#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

void PoolingLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob<float> *top = new Blob<float>(layer_param_.top(0));

  if (!(bottom->shape(1) && bottom->shape(2) && bottom->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();

  int in_h = bottom->shape(2), in_w = bottom->shape(3);
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
  std::cout << "Pooling Layer: " << Util::format_vector(bottom->shape(), " x ")
            << " input -> " << kernel_size_ << "x" << kernel_size_ << "_s"
            << stride_ << " -> " << Util::format_vector(top->shape(), " x ")
            << " output" << std::endl;
#endif
}

void PoolingLayer::Forward() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  Image::Pooling(bottom->shape(), bottom->data(), kernel_size_, stride_,
                 pool_type_, top->shape(), top->mutable_data());
}

void PoolingLayer::Release() {
  bottom_.clear();
  top_.clear();

  // std::cout << "Free PoolingLayer!" << std::endl;
}
