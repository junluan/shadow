#include "shadow/layers/dropout_layer.hpp"

void DropoutLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  std::stringstream out;
  out << layer_param_.name() << ": ("
      << Util::format_vector(bottom->shape(), ",") << ") -> ("
      << Util::format_vector(bottom->shape(), ",") << ")";
  DInfo(out.str());
}

void DropoutLayer::Forward() {}

void DropoutLayer::Release() {
  // std::cout << "Free DropoutLayer!" << std::endl;
}
