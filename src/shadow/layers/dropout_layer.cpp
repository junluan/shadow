#include "shadow/layers/dropout_layer.hpp"

DropoutLayer::DropoutLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob = new Blob();
  out_blob = new Blob();
}
DropoutLayer::~DropoutLayer() { ReleaseLayer(); }

void DropoutLayer::MakeLayer(Blob *blob) {
  if (!(blob->shape(1) && blob->shape(2) && blob->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  int in_num = blob->shape(1) * blob->shape(2) * blob->shape(3);
  int out_num = in_num;

#ifdef VERBOSE
  printf("Dropout Layer: %d input, %d output, %.1f probability\n", in_num,
         out_num, layer_param_.dropout_param().probability());
#endif
}

void DropoutLayer::ForwardLayer() { out_data_ = in_data_; }

#ifdef USE_CUDA
void DropoutLayer::CUDAForwardLayer() { cuda_out_data_ = cuda_in_data_; }
#endif

#ifdef USE_CL
void DropoutLayer::CLForwardLayer() { cl_out_data_ = cl_in_data_; }
#endif

void DropoutLayer::ReleaseLayer() {
  // std::cout << "Free DropoutLayer!" << std::endl;
}
