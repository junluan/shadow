#include "shadow/layers/connected_layer.hpp"
#include "shadow/util/blas.hpp"

void ConnectedLayer::Reshape() {
  num_output_ = layer_param_.connected_param().num_output();

  top_[0]->add_shape(bottom_[0]->shape(0));
  top_[0]->add_shape(num_output_);
  top_[0]->allocate_data(top_[0]->count());

  weights_ = new Blob<float>(bottom_[0]->num() * num_output_,
                             layer_name_ + " weights");
  biases_ = new Blob<float>(num_output_, layer_name_ + " biases");

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ConnectedLayer::Forward() {
  int batch = bottom_[0]->shape(0);
  int top_num = top_[0]->num(), bottom_num = bottom_[0]->num();
  for (int b = 0; b < batch; ++b) {
    Blas::BlasCopy(top_num, biases_->data(), 1, top_[0]->mutable_data(),
                   b * top_num, 1);
  }
  Blas::BlasSGemm(0, 0, batch, top_num, bottom_num, 1, bottom_[0]->data(),
                  bottom_num, weights_->data(), top_num, 1,
                  top_[0]->mutable_data(), 0, top_num);
}

void ConnectedLayer::Release() {
  bottom_.clear();
  top_.clear();

  weights_->clear();
  biases_->clear();

  // DInfo("Free ConnectedLayer!");
}
