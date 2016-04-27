#include "dropout_layer.hpp"

DropoutLayer::DropoutLayer(LayerType type) { layer_type_ = type; }
DropoutLayer::~DropoutLayer() { ReleaseLayer(); }

void DropoutLayer::MakeDropoutLayer(SizeParams params, float probability) {
  batch_ = params.batch;
  out_c_ = params.in_c;
  out_h_ = params.in_h;
  out_w_ = params.in_w;

  in_num_ = params.in_num;
  out_num_ = params.in_num;

#ifdef USE_CL
// cl_out_data_ = CL::CLMakeBuffer(batch_ * out_num_, CL_MEM_READ_WRITE, NULL);
#endif

#ifdef VERBOSE
  printf("Dropout Layer: %d input, %d output, %.1f probability\n", in_num_,
         out_num_, probability);
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
