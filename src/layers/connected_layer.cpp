#include "connected_layer.h"
#include "blas.h"
#include "kernel.h"

ConnectedLayer::ConnectedLayer(LayerType type) { layer_type_ = type; }
ConnectedLayer::~ConnectedLayer() { ReleaseLayer(); }

void ConnectedLayer::MakeConnectedLayer(SizeParams params, int out_num,
                                        std::string activation) {
  batch_ = params.batch;
  in_num_ = params.in_num;
  out_num_ = out_num;

  out_data_ = new float[batch_ * out_num_];

  weights_ = new float[in_num_ * out_num_];
  biases_ = new float[out_num_];

  activation_ = Activations::GetActivation(activation);

#ifdef USE_CUDA
  cuda_out_data_ = CUDA::CUDAMakeBuffer(batch_ * out_num_, NULL);
  cuda_weights_ = CUDA::CUDAMakeBuffer(in_num_ * out_num_, NULL);
  cuda_biases_ = CUDA::CUDAMakeBuffer(out_num_, NULL);
#endif

#ifdef USE_CL
  cl_out_data_ = CL::CLMakeBuffer(batch_ * out_num_, CL_MEM_READ_WRITE, NULL);
  cl_weights_ = CL::CLMakeBuffer(in_num_ * out_num_, CL_MEM_READ_ONLY, NULL);
  cl_biases_ = CL::CLMakeBuffer(out_num_, CL_MEM_READ_ONLY, NULL);
#endif

#ifdef VERBOSE
  printf("Connected Layer: %d input, %d output\n", in_num_, out_num_);
#endif
}

void ConnectedLayer::ForwardLayer() {
  for (int b = 0; b < batch_; ++b) {
    Blas::BlasCopy(out_num_, biases_, 1, out_data_ + b * out_num_, 1);
  }
  Blas::BlasSGemm(0, 0, batch_, out_num_, in_num_, 1, in_data_, in_num_,
                  weights_, out_num_, 1, out_data_, out_num_);
  Activations::ActivateArray(batch_ * out_num_, activation_, out_data_);
}

#ifdef USE_CUDA
void ConnectedLayer::CUDAForwardLayer() {
  Kernel::CUDABiasOutput(cuda_biases_, batch_, out_num_, 1, cuda_out_data_);
  Blas::CUDABlasSGemm(0, 0, batch_, out_num_, in_num_, 1, cuda_in_data_,
                      in_num_, cuda_weights_, out_num_, 1, cuda_out_data_, 0,
                      out_num_);
  Kernel::CUDAActivateArray(batch_ * out_num_, activation_, cuda_out_data_);
}
#endif

#ifdef USE_CL
void ConnectedLayer::CLForwardLayer() {
  Kernel::CLBiasOutput(cl_biases_, batch_, out_num_, 1, cl_out_data_);
  Blas::CLBlasSGemm(0, 0, batch_, out_num_, in_num_, 1, cl_in_data_, in_num_,
                    cl_weights_, out_num_, 1, cl_out_data_, 0, out_num_);
  Kernel::CLActivateArray(batch_ * out_num_, activation_, cl_out_data_);
}
#endif

float *ConnectedLayer::GetOutData() {
#ifdef USE_CUDA
  CUDA::CUDAReadBuffer(batch_ * out_num_, cuda_out_data_, out_data_);

#else
#ifdef USE_CL
  CL::CLReadBuffer(batch_ * out_num_, cl_out_data_, out_data_);
#endif
#endif
  return out_data_;
}

void ConnectedLayer::ReleaseLayer() {
  if (out_data_ != NULL)
    delete[] out_data_;
  if (weights_ != NULL)
    delete[] weights_;
  if (biases_ != NULL)
    delete[] biases_;

#ifdef USE_CUDA
  if (cuda_out_data_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_out_data_);
  if (cuda_weights_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_weights_);
  if (cuda_biases_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_biases_);
#endif

#ifdef USE_CL
  if (cl_out_data_ != NULL)
    CL::CLReleaseBuffer(cl_out_data_);
  if (cl_weights_ != NULL)
    CL::CLReleaseBuffer(cl_weights_);
  if (cl_biases_ != NULL)
    CL::CLReleaseBuffer(cl_biases_);
#endif
  // std::cout << "Free ConnectedLayer!" << std::endl;
}
