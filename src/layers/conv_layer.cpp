#include "conv_layer.h"
#include "blas.h"
#include "image.h"

int convolutional_out_size(int s, int size, int pad, int stride) {
  return (s + 2 * pad - size) / stride + 1;
}

ConvLayer::ConvLayer(LayerType type) { layer_type_ = type; }
ConvLayer::~ConvLayer() {}

void ConvLayer::MakeConvLayer(SizeParams params, int out_num, int ksize,
                              int stride, int pad, std::string activation) {
  batch_ = params.batch;
  in_c_ = params.in_c;
  in_h_ = params.in_h;
  in_w_ = params.in_w;
  out_c_ = out_num;
  out_h_ = convolutional_out_size(in_h_, ksize, pad, stride);
  out_w_ = convolutional_out_size(in_w_, ksize, pad, stride);

  ksize_ = ksize;
  stride_ = stride;
  pad_ = pad;

  in_num_ = in_c_ * in_h_ * in_w_;
  out_num_ = out_c_ * out_h_ * out_w_;
  out_map_size_ = out_h_ * out_w_;
  kernel_num_ = ksize_ * ksize_ * in_c_;

  out_data_ = new float[batch_ * out_num_];

  filters_ = new float[kernel_num_ * out_c_];
  biases_ = new float[out_c_];
  col_image_ = new float[out_map_size_ * kernel_num_];

  activation_ = Activations::GetActivation(activation);

#ifdef USE_CL
  cl_out_data_ = CL::CLMakeBuffer(batch_ * out_num_, CL_MEM_READ_WRITE, NULL);
  cl_filters_ = CL::CLMakeBuffer(kernel_num_ * out_c_, CL_MEM_READ_ONLY, NULL);
  cl_biases_ = CL::CLMakeBuffer(out_c_, CL_MEM_READ_ONLY, NULL);
  cl_col_image_ =
      CL::CLMakeBuffer(out_map_size_ * kernel_num_, CL_MEM_READ_WRITE, NULL);
#endif

#ifdef VERBOSE
  printf(
      "Convolutional Layer: %d x %d x %d input -> %d_%dx%d_s%d_p%d filters -> "
      "%d x %d x %d "
      "output\n",
      in_h_, in_w_, in_c_, out_c_, ksize_, ksize_, stride_, pad_, out_h_,
      out_w_, out_c_);
#endif
}

void BiasOutput(float *out_data, float *biases, int batch, int n, int size) {
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < size; ++j) {
        out_data[(b * n + i) * size + j] = biases[i];
      }
    }
  }
}

void ConvLayer::ForwardLayer() {
  BiasOutput(out_data_, biases_, batch_, out_c_, out_map_size_);
  for (int b = 0; b < batch_; ++b) {
    Image::Im2Col(in_data_ + b * in_num_, in_c_, in_h_, in_w_, ksize_,
                         stride_, pad_, out_h_, out_w_, col_image_);
    Blas::BlasSGemm(0, 0, out_c_, out_map_size_, kernel_num_, 1, filters_,
                    kernel_num_, col_image_, out_map_size_, 1,
                    out_data_ + b * out_num_, out_map_size_);
  }
  Activations::ActivateArray(batch_ * out_num_, activation_, out_data_);
}

#ifdef USE_CL
void ConvLayer::CLForwardLayer() {
  CL::CLBiasOutput(cl_biases_, batch_, out_c_, out_map_size_, cl_out_data_);
  for (int b = 0; b < batch_; ++b) {
    Image::CLIm2Col(cl_in_data_, b * in_num_, in_c_, in_h_, in_w_,
                           ksize_, stride_, pad_, out_h_, out_w_,
                           cl_col_image_);
    Blas::CLBlasSGemm(0, 0, out_c_, out_map_size_, kernel_num_, 1, cl_filters_,
                      kernel_num_, cl_col_image_, out_map_size_, 1,
                      cl_out_data_, b * out_num_, out_map_size_);
  }
  Activations::CLActivateArray(batch_ * out_num_, activation_, cl_out_data_);
}
#endif

void ConvLayer::ReleaseLayer() {
  if (out_data_ != NULL)
    delete[] out_data_;
  if (filters_ != NULL)
    delete[] filters_;
  if (biases_ != NULL)
    delete[] biases_;
  if (col_image_ != NULL)
    delete[] col_image_;

#ifdef USE_CL
  if (cl_out_data_ != NULL)
    CL::CLReleaseBuffer(cl_out_data_);
  if (cl_filters_ != NULL)
    CL::CLReleaseBuffer(cl_filters_);
  if (cl_biases_ != NULL)
    CL::CLReleaseBuffer(cl_biases_);
  if (cl_col_image_ != NULL)
    CL::CLReleaseBuffer(cl_col_image_);
#endif
  // std::cout << "Free ConvLayer!" << std::endl;
}
