#include "connected_layer.hpp"
#include "blas.hpp"

ConnectedLayer::ConnectedLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob = new shadow::Blob();
  out_blob = new shadow::Blob();
}
ConnectedLayer::~ConnectedLayer() { ReleaseLayer(); }

void ConnectedLayer::MakeLayer(shadow::BlobShape *shape) {
  if (!(shape->dim(1) && shape->dim(2) && shape->dim(3)))
    Fatal("Channel, height and width must greater than zero.");

  layer_type_ = shadow::LayerType::Connected;
  num_output_ = layer_param_.connected_param().num_output();
  activate_ = layer_param_.connected_param().activate();

  int batch = shape->dim(0);

  int in_num = shape->dim(1) * shape->dim(2) * shape->dim(3);
  int out_num = num_output_;

  *in_blob->mutable_shape() = *shape;
  shape->set_dim(1, num_output_);
  shape->set_dim(2, 1);
  shape->set_dim(3, 1);
  *out_blob->mutable_shape() = *shape;

  in_blob->set_num(in_num);
  out_blob->set_num(out_num);
  in_blob->set_count(batch * in_num);
  out_blob->set_count(batch * out_num);

  out_data_ = new float[out_blob->count()];

  weights_ = new float[in_num * out_num];
  biases_ = new float[out_num];

#ifdef USE_CUDA
  cuda_out_data_ = CUDA::CUDAMakeBuffer(out_blob->count(), NULL);
  cuda_weights_ = CUDA::CUDAMakeBuffer(in_num * out_num, NULL);
  cuda_biases_ = CUDA::CUDAMakeBuffer(out_num, NULL);
#endif

#ifdef USE_CL
  cl_out_data_ = CL::CLMakeBuffer(out_blob->count(), CL_MEM_READ_WRITE, NULL);
  cl_weights_ = CL::CLMakeBuffer(in_num * out_num, CL_MEM_READ_ONLY, NULL);
  cl_biases_ = CL::CLMakeBuffer(out_num, CL_MEM_READ_ONLY, NULL);
#endif

#ifdef VERBOSE
  printf("Connected Layer: %d input, %d output\n", in_num, out_num);
#endif
}

void ConnectedLayer::ForwardLayer() {
  int batch = in_blob->shape().dim(0);
  for (int b = 0; b < batch; ++b) {
    Blas::BlasCopy(out_blob->num(), biases_, 1, out_data_ + b * out_blob->num(),
                   1);
  }
  Blas::BlasSGemm(0, 0, batch, out_blob->num(), in_blob->num(), 1, in_data_,
                  in_blob->num(), weights_, out_blob->num(), 1, out_data_,
                  out_blob->num());
  Activations::ActivateArray(out_blob->count(), activate_, out_data_);
}

#ifdef USE_CUDA
void ConnectedLayer::CUDAForwardLayer() {
  int batch = in_blob->shape().dim(0);
  Kernel::CUDABiasOutput(cuda_biases_, batch, out_blob->num(), 1,
                         cuda_out_data_);
  Blas::CUDABlasSGemm(0, 0, batch, out_blob->num(), in_blob->num(), 1,
                      cuda_in_data_, in_blob->num(), cuda_weights_,
                      out_blob->num(), 1, cuda_out_data_, 0, out_blob->num());
  Kernel::CUDAActivateArray(out_blob->count(), activate_, cuda_out_data_);
}
#endif

#ifdef USE_CL
void ConnectedLayer::CLForwardLayer() {
  int batch = in_blob->shape().dim(0);
  Kernel::CLBiasOutput(cl_biases_, batch, out_blob->num(), 1, cl_out_data_);
  Blas::CLBlasSGemm(0, 0, batch, out_blob->num(), in_blob->num(), 1,
                    cl_in_data_, in_blob->num(), cl_weights_, out_blob->num(),
                    1, cl_out_data_, 0, out_blob->num());
  Kernel::CLActivateArray(out_blob->count(), activate_, cl_out_data_);
}
#endif

float *ConnectedLayer::GetOutData() {
#ifdef USE_CUDA
  CUDA::CUDAReadBuffer(out_blob->count(), cuda_out_data_, out_data_);
#else
#ifdef USE_CL
  CL::CLReadBuffer(out_blob->count(), cl_out_data_, out_data_);
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
