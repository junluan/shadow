#include "dropout_layer.hpp"

DropoutLayer::DropoutLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob = new shadow::Blob();
  out_blob = new shadow::Blob();
}
DropoutLayer::~DropoutLayer() { ReleaseLayer(); }

void DropoutLayer::MakeLayer(shadow::BlobShape *shape) {
  if (!(shape->dim(1) && shape->dim(2) && shape->dim(3)))
    Fatal("Channel, height and width must greater than zero.");

  layer_type_ = shadow::LayerType::Dropout;

  int in_num = shape->dim(1) * shape->dim(2) * shape->dim(3);
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
