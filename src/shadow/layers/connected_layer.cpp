#include "shadow/layers/connected_layer.hpp"
#include "shadow/util/blas.hpp"

void ConnectedLayer::Reshape() {
  num_output_ = layer_param_.connected_param().num_output();

  VecInt top_shape;
  top_shape.push_back(bottoms_[0]->shape(0));
  top_shape.push_back(num_output_);
  tops_[0]->reshape(top_shape);

  if (blobs_.size() == 0) {
    blobs_.push_back(new Blob<float>(num_output_ * bottoms_[0]->num()));
    blobs_.push_back(new Blob<float>(num_output_));
  }

  biases_multiplier_.reshape(top_shape[0]);
  Blas::Set(top_shape[0], 1, biases_multiplier_.mutable_data(), 0);

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ConnectedLayer::Forward() {
  int batch = bottoms_[0]->shape(0);
  int top_num = tops_[0]->num(), bottom_num = bottoms_[0]->num();
  if (batch == 1) {
    Blas::BlasSgemv(0, num_output_, bottom_num, 1, blobs_[0]->data(), 0,
                    bottoms_[0]->data(), 0, 0, tops_[0]->mutable_data(), 0);
    Blas::BlasSaxpy(num_output_, 1, blobs_[1]->data(), 0,
                    tops_[0]->mutable_data(), 0);
  } else {
    Blas::BlasSgemm(0, 1, batch, top_num, bottom_num, 1, bottoms_[0]->data(), 0,
                    blobs_[0]->data(), 0, 0, tops_[0]->mutable_data(), 0);
    Blas::BlasSgemm(0, 0, batch, num_output_, 1, 1, biases_multiplier_.data(),
                    0, blobs_[1]->data(), 0, 1, tops_[0]->mutable_data(), 0);
  }
}

void ConnectedLayer::Release() {
  bottoms_.clear();
  tops_.clear();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->clear();
  }
  blobs_.clear();

  biases_multiplier_.clear();

  // DInfo("Free ConnectedLayer!");
}
