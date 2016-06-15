#include "shadow/layers/connected_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"

ConnectedLayer::ConnectedLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  weights_ = new Blob();
  biases_ = new Blob();
}
ConnectedLayer::~ConnectedLayer() { Release(); }

void ConnectedLayer::Setup(VecBlob *blobs) {
  Blob *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob *top = new Blob(layer_param_.top(0));

  if (!(bottom->shape(1) && bottom->shape(2) && bottom->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  num_output_ = layer_param_.connected_param().num_output();
  activate_ = layer_param_.connected_param().activate();

  int in_num = bottom->shape(1) * bottom->shape(2) * bottom->shape(3);
  int out_num = num_output_;

  *top->mutable_shape() = bottom->shape();
  top->set_shape(1, num_output_);
  top->set_shape(2, 1);
  top->set_shape(3, 1);

  top->allocate_data(top->count());

  bottom_.push_back(bottom);
  top_.push_back(top);

  blobs->push_back(top);

  weights_->allocate_data(in_num * out_num);
  biases_->allocate_data(out_num);

#if defined(VERBOSE)
  printf("Connected Layer: %d input, %d output\n", in_num, out_num);
#endif
}

void ConnectedLayer::Forward() {
  Blob *bottom = bottom_.at(0);
  Blob *top = top_.at(0);

  int batch = bottom->shape(0);
  BType *out_data = top->mutable_data();
  for (int b = 0; b < batch; ++b) {
    Blas::BlasCopy(top->num(), biases_->data(), 1, out_data + b * top->num(),
                   1);
  }
  Blas::BlasSGemm(0, 0, batch, top->num(), bottom->num(), 1, bottom->data(),
                  bottom->num(), weights_->data(), top->num(), 1, out_data, 0,
                  top->num());
  Activations::ActivateArray(top->count(), activate_, out_data);
}

void ConnectedLayer::Release() {
  bottom_.clear();
  top_.clear();

  weights_->clear();
  biases_->clear();

  // std::cout << "Free ConnectedLayer!" << std::endl;
}
