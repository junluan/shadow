#include "pooling_layer.h"
#include "image.h"
#include "kernel.h"

PoolingLayer::PoolingLayer(LayerType type) { layer_type_ = type; }
PoolingLayer::~PoolingLayer() { ReleaseLayer(); }

void PoolingLayer::MakePoolingLayer(SizeParams params, int ksize, int stride,
                                    std::string pool_type) {
  batch_ = params.batch;
  in_c_ = params.in_c;
  in_h_ = params.in_h;
  in_w_ = params.in_w;
  out_c_ = in_c_;
  out_h_ = (in_h_ - ksize) / stride + 1;
  out_w_ = (in_w_ - ksize) / stride + 1;

  ksize_ = ksize;
  stride_ = stride;
  pool_type_ = pool_type.compare("Max") ? kAve : kMax;

  in_num_ = in_c_ * in_h_ * in_w_;
  out_num_ = out_c_ * out_h_ * out_w_;
  out_data_ = new float[batch_ * out_num_];

#ifdef USE_CUDA
  cuda_out_data_ = CUDA::CUDAMakeBuffer(batch_ * out_num_, NULL);
#endif

#ifdef USE_CL
  cl_out_data_ = CL::CLMakeBuffer(batch_ * out_num_, CL_MEM_READ_WRITE, NULL);
#endif

#ifdef VERBOSE
  printf("Maxpool Layer: %d x %d x %d input -> %dx%d_s%d -> "
         "%d x %d x %d "
         "output\n",
         in_h_, in_w_, in_c_, ksize_, ksize_, stride_, out_h_, out_w_, out_c_);
#endif
}

void PoolingLayer::ForwardLayer() {
  Image::Pooling(in_data_, batch_, in_c_, in_h_, in_w_, ksize_, stride_, out_h_,
                 out_w_, pool_type_, out_data_);
}

#ifdef USE_CUDA
void PoolingLayer::CUDAForwardLayer() {
  Kernel::CUDAPooling(cuda_in_data_, batch_, in_c_, in_h_, in_w_, ksize_,
                      stride_, out_h_, out_w_, pool_type_, cuda_out_data_);
}
#endif

#ifdef USE_CL
void PoolingLayer::CLForwardLayer() {
  Kernel::CLPooling(cl_in_data_, batch_, in_c_, in_h_, in_w_, ksize_, stride_,
                    out_h_, out_w_, pool_type_, cl_out_data_);
}
#endif

void PoolingLayer::ReleaseLayer() {
  if (out_data_ != NULL)
    delete[] out_data_;

#ifdef USE_CUDA
  if (cuda_out_data_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_out_data_);
#endif

#ifdef USE_CL
  if (cl_out_data_ != NULL)
    CL::CLReleaseBuffer(cl_out_data_);
#endif
  // std::cout << "Free PoolingLayer!" << std::endl;
}
