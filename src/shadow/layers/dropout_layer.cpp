#include "shadow/layers/dropout_layer.hpp"

void DropoutLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

#if defined(VERBOSE)
  std::cout << "Dropout Layer: " << format_vector(bottom->shape(), " x ")
            << " input -> " << format_vector(bottom->shape(), " x ")
            << " output" << std::endl;
#endif
}

void DropoutLayer::Forward() {}

void DropoutLayer::Release() {
  // std::cout << "Free DropoutLayer!" << std::endl;
}
