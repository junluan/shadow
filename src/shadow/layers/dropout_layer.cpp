#include "shadow/layers/dropout_layer.hpp"

void DropoutLayer::Setup(VecBlob *blobs) {
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
}

void DropoutLayer::Reshape() {
  const Blob<float> *bottom = bottom_.at(0);

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(bottom->shape(), ",", "(", ")");
  DInfo(out.str());
}

void DropoutLayer::Forward() {}

void DropoutLayer::Release() {
  bottom_.clear();
  top_.clear();

  // std::cout << "Free DropoutLayer!" << std::endl;
}
