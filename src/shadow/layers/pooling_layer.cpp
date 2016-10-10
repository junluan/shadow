#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

void PoolingLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom != nullptr) {
    if (bottom->num() && bottom->num_axes() == 4) {
      bottom_.push_back(bottom);
    } else {
      Fatal(layer_name_ + ": bottom blob(" + layer_param_.bottom(0) +
            Util::format_vector(bottom->shape(), ",", "(", ")") +
            ") dimension mismatch!");
    }
  } else {
    Fatal(layer_name_ + ": bottom blob(" + layer_param_.bottom(0) +
          ") not exist!");
  }

  for (int i = 0; i < layer_param_.top_size(); ++i) {
    Blob<float> *top = new Blob<float>(layer_param_.top(i));
    top_.push_back(top);
    blobs->push_back(top);
  }

  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();
}

void PoolingLayer::Reshape() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  int in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_h = (in_h - kernel_size_) / stride_ + 1;
  int out_w = (in_w - kernel_size_) / stride_ + 1;

  *top->mutable_shape() = bottom->shape();
  top->set_shape(2, out_h);
  top->set_shape(3, out_w);
  top->allocate_data(top->count());

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
      << kernel_size_ << "x" << kernel_size_ << "_s" << stride_ << " -> "
      << Util::format_vector(top->shape(), ",", "(", ")");
  DInfo(out.str());
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
