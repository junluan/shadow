#include "shadow/layers/connected_layer.hpp"
#include "shadow/util/blas.hpp"

void ConnectedLayer::Reshape() {
  num_output_ = layer_param_.connected_param().num_output();

  VecInt top_shape;
  top_shape.push_back(bottom_[0]->shape(0));
  top_shape.push_back(num_output_);
  top_[0]->reshape(top_shape);

  weights_.reshape(num_output_, bottom_[0]->num());
  biases_.reshape(num_output_);
  biases_multiplier_.reshape(top_shape[0]);
  Blas::Set(top_shape[0], 1, biases_multiplier_.mutable_data(), 0);

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ConnectedLayer::Forward() {
  int batch = bottom_[0]->shape(0);
  int top_num = top_[0]->num(), bottom_num = bottom_[0]->num();
  if (batch == 1) {
    Blas::BlasSgemv(1, bottom_num, num_output_, 1, weights_.data(), 0,
                    bottom_[0]->data(), 0, 0, top_[0]->mutable_data(), 0);
    Blas::BlasSaxpy(num_output_, 1, biases_.data(), 0, top_[0]->mutable_data(),
                    0);
  } else {
    Blas::BlasSgemm(0, 0, batch, top_num, bottom_num, 1, bottom_[0]->data(), 0,
                    weights_.data(), 0, 0, top_[0]->mutable_data(), 0);
    Blas::BlasSgemm(0, 0, batch, num_output_, 1, 1, biases_multiplier_.data(),
                    0, biases_.data(), 0, 1, top_[0]->mutable_data(), 0);
  }
}

void ConnectedLayer::Release() {
  bottom_.clear();
  top_.clear();

  weights_.clear();
  biases_.clear();
  biases_multiplier_.clear();

  // DInfo("Free ConnectedLayer!");
}
