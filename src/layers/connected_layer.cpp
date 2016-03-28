#include "connected_layer.h"
#include "blas.h"

ConnectedLayer::ConnectedLayer(LayerType type) { layer_type_ = type; }
ConnectedLayer::~ConnectedLayer() {}

void ConnectedLayer::MakeConnectedLayer(SizeParams params, int out_num,
                                        std::string activation) {
  batch_ = params.batch;
  in_num_ = params.in_num;
  out_num_ = out_num;

  out_data_ = new float[batch_ * out_num_];

  weights_ = new float[in_num_ * out_num_];
  biases_ = new float[out_num_];

  activation_ = Activations::GetActivation(activation);

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

#ifdef USE_CL
void ConnectedLayer::CLForwardLayer() {
  CL::CLBiasOutput(cl_biases_, batch_, out_num_, 1, cl_out_data_);
  Blas::CLBlasSGemm(0, 0, batch_, out_num_, in_num_, 1, cl_in_data_, in_num_,
                    cl_weights_, out_num_, 1, cl_out_data_, 0, out_num_);
  Activations::CLActivateArray(batch_ * out_num_, activation_, cl_out_data_);
}
#endif

void ConnectedLayer::ReleaseLayer() {
  if (out_data_ != NULL)
    delete[] out_data_;
  if (weights_ != NULL)
    delete[] weights_;
  if (biases_ != NULL)
    delete[] biases_;

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
