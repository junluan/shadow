#include "shadow/layers/dropout_layer.hpp"

void DropoutLayer::Setup(VecBlob *blobs) {
  Blob *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));

  if (!(bottom->shape(1) && bottom->shape(2) && bottom->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  int in_num = bottom->shape(1) * bottom->shape(2) * bottom->shape(3);
  int out_num = in_num;

#if defined(VERBOSE)
  printf("Dropout Layer: %d input, %d output, %.1f probability\n", in_num,
         out_num, layer_param_.dropout_param().probability());
#endif
}

void DropoutLayer::Forward() {}

void DropoutLayer::Release() {
  // std::cout << "Free DropoutLayer!" << std::endl;
}
