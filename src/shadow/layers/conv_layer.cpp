#include "shadow/layers/conv_layer.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

inline int convolutional_out_size(int s, int size, int pad, int stride) {
  return (s + 2 * pad - size) / stride + 1;
}

ConvLayer::ConvLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob = new Blob();
  out_blob = new Blob();
}
ConvLayer::~ConvLayer() { ReleaseLayer(); }

void ConvLayer::MakeLayer(Blob *blob) {
  if (!(blob->shape(1) && blob->shape(2) && blob->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  num_output_ = layer_param_.convolution_param().num_output();
  kernel_size_ = layer_param_.convolution_param().kernel_size();
  stride_ = layer_param_.convolution_param().stride();
  pad_ = layer_param_.convolution_param().pad();
  activate_ = layer_param_.convolution_param().activate();

  int batch = blob->shape(0);
  int in_c = blob->shape(1), in_h = blob->shape(2), in_w = blob->shape(3);
  int out_c = num_output_;
  int out_h = convolutional_out_size(in_h, kernel_size_, pad_, stride_);
  int out_w = convolutional_out_size(in_w, kernel_size_, pad_, stride_);

  int in_num = in_c * in_h * in_w;
  int out_num = out_c * out_h * out_w;

  *in_blob->mutable_shape() = blob->shape();
  blob->set_shape(1, out_c);
  blob->set_shape(2, out_h);
  blob->set_shape(3, out_w);
  *out_blob->mutable_shape() = blob->shape();

  in_blob->set_num(in_num);
  out_blob->set_num(out_num);
  in_blob->set_count(batch * in_num);
  out_blob->set_count(batch * out_num);

  out_map_size_ = out_h * out_w;
  kernel_num_ = kernel_size_ * kernel_size_ * in_c;

  out_data_ = new float[out_blob->count()];

  filters_ = new float[kernel_num_ * out_c];
  biases_ = new float[out_c];
  col_image_ = new float[out_map_size_ * kernel_num_];

#ifdef USE_CUDA
  cuda_out_data_ = CUDA::CUDAMakeBuffer(out_blob->count(), NULL);
  cuda_filters_ = CUDA::CUDAMakeBuffer(kernel_num_ * out_c, NULL);
  cuda_biases_ = CUDA::CUDAMakeBuffer(out_c, NULL);
  cuda_col_image_ = CUDA::CUDAMakeBuffer(out_map_size_ * kernel_num_, NULL);
#endif

#ifdef USE_CL
  cl_out_data_ = CL::CLMakeBuffer(out_blob->count(), CL_MEM_READ_WRITE, NULL);
  cl_filters_ = CL::CLMakeBuffer(kernel_num_ * out_c, CL_MEM_READ_ONLY, NULL);
  cl_biases_ = CL::CLMakeBuffer(out_c, CL_MEM_READ_ONLY, NULL);
  cl_col_image_ =
      CL::CLMakeBuffer(out_map_size_ * kernel_num_, CL_MEM_READ_WRITE, NULL);
#endif

#ifdef VERBOSE
  printf(
      "Convolutional Layer: %d x %d x %d input -> %d_%dx%d_s%d_p%d filters -> "
      "%d x %d x %d output\n",
      in_c, in_h, in_w, out_c, kernel_size_, kernel_size_, stride_, pad_, out_c,
      out_h, out_w);
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
  int batch = in_blob->shape(0), in_c = in_blob->shape(1);
  int in_h = in_blob->shape(2), in_w = in_blob->shape(3);
  int out_c = out_blob->shape(1);
  int out_h = out_blob->shape(2), out_w = out_blob->shape(3);
  BiasOutput(out_data_, biases_, batch, out_c, out_map_size_);
  for (int b = 0; b < batch; ++b) {
    Image::Im2Col(in_data_ + b * in_blob->num(), in_c, in_h, in_w, kernel_size_,
                  stride_, pad_, out_h, out_w, col_image_);
    Blas::BlasSGemm(0, 0, out_c, out_map_size_, kernel_num_, 1, filters_,
                    kernel_num_, col_image_, out_map_size_, 1,
                    out_data_ + b * out_blob->num(), out_map_size_);
  }
  Activations::ActivateArray(out_blob->count(), activate_, out_data_);
}

#ifdef USE_CUDA
void ConvLayer::CUDAForwardLayer() {
  int batch = in_blob->shape(0), in_c = in_blob->shape(1);
  int in_h = in_blob->shape(2), in_w = in_blob->shape(3);
  int out_c = out_blob->shape(1);
  int out_h = out_blob->shape(2), out_w = out_blob->shape(3);
  Kernel::CUDABiasOutput(cuda_biases_, batch, out_c, out_map_size_,
                         cuda_out_data_);
  for (int b = 0; b < batch; ++b) {
    Kernel::CUDAIm2Col(cuda_in_data_, b * in_blob->num(), in_c, in_h, in_w,
                       kernel_size_, stride_, pad_, out_h, out_w,
                       cuda_col_image_);
    Blas::CUDABlasSGemm(0, 0, out_c, out_map_size_, kernel_num_, 1,
                        cuda_filters_, kernel_num_, cuda_col_image_,
                        out_map_size_, 1, cuda_out_data_, b * out_blob->num(),
                        out_map_size_);
  }
  Kernel::CUDAActivateArray(out_blob->count(), activate_, cuda_out_data_);
}
#endif

#ifdef USE_CL
void ConvLayer::CLForwardLayer() {
  int batch = in_blob->shape().dim(0), in_c = in_blob->shape().dim(1);
  int in_h = in_blob->shape().dim(2), in_w = in_blob->shape().dim(3);
  int out_c = out_blob->shape().dim(1);
  int out_h = out_blob->shape().dim(2), out_w = out_blob->shape().dim(3);
  Kernel::CLBiasOutput(cl_biases_, batch, out_c, out_map_size_, cl_out_data_);
  for (int b = 0; b < batch; ++b) {
    Kernel::CLIm2Col(cl_in_data_, b * in_blob->num(), in_c, in_h, in_w,
                     kernel_size_, stride_, pad_, out_h, out_w, cl_col_image_);
    Blas::CLBlasSGemm(0, 0, out_c, out_map_size_, kernel_num_, 1, cl_filters_,
                      kernel_num_, cl_col_image_, out_map_size_, 1,
                      cl_out_data_, b * out_blob->num(), out_map_size_);
  }
  Kernel::CLActivateArray(out_blob->count(), activate_, cl_out_data_);
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

#ifdef USE_CUDA
  if (cuda_out_data_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_out_data_);
  if (cuda_filters_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_filters_);
  if (cuda_biases_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_biases_);
  if (cuda_col_image_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_col_image_);
#endif

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
