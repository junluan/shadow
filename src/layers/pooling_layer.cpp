#include "pooling_layer.hpp"
#include "image.hpp"

PoolingLayer::PoolingLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob = new shadow::Blob();
  out_blob = new shadow::Blob();
}
PoolingLayer::~PoolingLayer() { ReleaseLayer(); }

void PoolingLayer::MakeLayer(shadow::BlobShape *shape) {
  if (!(shape->dim(1) && shape->dim(2) && shape->dim(3)))
    Fatal("Channel, height and width must greater than zero.");

  layer_type_ = shadow::LayerType::Pooling;
  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();

  int batch = shape->dim(0);
  int in_c = shape->dim(1), in_h = shape->dim(2), in_w = shape->dim(3);
  int out_c = in_c;
  int out_h = (in_h - kernel_size_) / stride_ + 1;
  int out_w = (in_w - kernel_size_) / stride_ + 1;

  int in_num = in_c * in_h * in_w;
  int out_num = out_c * out_h * out_w;

  *in_blob->mutable_shape() = *shape;
  shape->set_dim(2, out_h);
  shape->set_dim(3, out_w);
  *out_blob->mutable_shape() = *shape;

  in_blob->set_num(in_num);
  out_blob->set_num(out_num);
  in_blob->set_count(batch * in_num);
  out_blob->set_count(batch * out_num);

  out_data_ = new float[out_blob->count()];

#ifdef USE_CUDA
  cuda_out_data_ = CUDA::CUDAMakeBuffer(out_blob->count(), NULL);
#endif

#ifdef USE_CL
  cl_out_data_ = CL::CLMakeBuffer(batch_ * out_num_, CL_MEM_READ_WRITE, NULL);
#endif

#ifdef VERBOSE
  printf(
      "Maxpool Layer: %d x %d x %d input -> %dx%d_s%d -> %d x %d x %d output\n",
      in_c, in_h, in_w, kernel_size_, kernel_size_, stride_, out_c, out_h,
      out_w);
#endif
}

void PoolingLayer::ForwardLayer() {
  int batch = in_blob->shape().dim(0), in_c = in_blob->shape().dim(1);
  int in_h = in_blob->shape().dim(2), in_w = in_blob->shape().dim(3);
  int out_h = out_blob->shape().dim(2), out_w = out_blob->shape().dim(3);
  Image::Pooling(in_data_, batch, in_c, in_h, in_w, kernel_size_, stride_,
                 out_h, out_w, pool_type_, out_data_);
}

#ifdef USE_CUDA
void PoolingLayer::CUDAForwardLayer() {
  int batch = in_blob->shape().dim(0), in_c = in_blob->shape().dim(1);
  int in_h = in_blob->shape().dim(2), in_w = in_blob->shape().dim(3);
  int out_h = out_blob->shape().dim(2), out_w = out_blob->shape().dim(3);
  Kernel::CUDAPooling(cuda_in_data_, batch, in_c, in_h, in_w, kernel_size_,
                      stride_, out_h, out_w, pool_type_, cuda_out_data_);
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
