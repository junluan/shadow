#include "shadow/layers/dropout_layer.hpp"

DropoutLayer::DropoutLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob_ = new Blob<BType>();
  out_blob_ = new Blob<BType>();
}
DropoutLayer::~DropoutLayer() { ReleaseLayer(); }

void DropoutLayer::MakeLayer(Blob<BType> *blob) {
  if (!(blob->shape(1) && blob->shape(2) && blob->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  int in_num = blob->shape(1) * blob->shape(2) * blob->shape(3);
  int out_num = in_num;

#if defined(VERBOSE)
  printf("Dropout Layer: %d input, %d output, %.1f probability\n", in_num,
         out_num, layer_param_.dropout_param().probability());
#endif
}

void DropoutLayer::ForwardLayer() { out_blob_ = in_blob_; }

void DropoutLayer::ReleaseLayer() {
  // std::cout << "Free DropoutLayer!" << std::endl;
}
