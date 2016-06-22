#include "shadow/layers/connected_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"

void ConnectedLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob<float> *top = new Blob<float>(layer_param_.top(0));

  num_output_ = layer_param_.connected_param().num_output();
  activate_ = layer_param_.connected_param().activate();

  int in_num = bottom->num();
  int out_num = num_output_;

  top->add_shape(bottom->shape(0));
  top->add_shape(num_output_);

  top->allocate_data(top->count());

  bottom_.push_back(bottom);
  top_.push_back(top);

  blobs->push_back(top);

  weights_ = new Blob<float>(in_num * out_num);
  biases_ = new Blob<float>(out_num);

#if defined(VERBOSE)
  std::cout << "Connected Layer: " << format_vector(bottom->shape(), " x ")
            << " input -> " << format_vector(top->shape(), " x ") << " output"
            << std::endl;
#endif
}

void ConnectedLayer::Forward() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  int batch = bottom->shape(0);
  for (int b = 0; b < batch; ++b) {
    Blas::BlasCopy(top->num(), biases_->data(), 1, top->mutable_data(),
                   b * top->num(), 1);
  }
  Blas::BlasSGemm(0, 0, batch, top->num(), bottom->num(), 1, bottom->data(),
                  bottom->num(), weights_->data(), top->num(), 1,
                  top->mutable_data(), 0, top->num());
  Activations::ActivateArray(top->count(), activate_, top->mutable_data());
}

void ConnectedLayer::Release() {
  bottom_.clear();
  top_.clear();

  weights_->clear();
  biases_->clear();

  // std::cout << "Free ConnectedLayer!" << std::endl;
}
