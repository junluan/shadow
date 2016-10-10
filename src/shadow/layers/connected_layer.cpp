#include "shadow/layers/connected_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"

void ConnectedLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom != nullptr) {
    if (bottom->num()) {
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

  num_output_ = layer_param_.connected_param().num_output();
  activate_ = layer_param_.connected_param().activate();
}

void ConnectedLayer::Reshape() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  top->add_shape(bottom->shape(0));
  top->add_shape(num_output_);
  top->allocate_data(top->count());

  weights_ =
      new Blob<float>(bottom->num() * num_output_, layer_name_ + " weights");
  biases_ = new Blob<float>(num_output_, layer_name_ + " biases");

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top->shape(), ",", "(", ")");
  DInfo(out.str());
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
