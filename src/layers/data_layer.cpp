#include "data_layer.hpp"

DataLayer::DataLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob = new shadow::Blob();
  out_blob = new shadow::Blob();
}
DataLayer::~DataLayer() { ReleaseLayer(); }

void DataLayer::MakeLayer(shadow::BlobShape *shape) {
  if (!(shape->dim(1) && shape->dim(2) && shape->dim(3)))
    Fatal("Channel, height and width must greater than zero.");

  layer_type_ = shadow::LayerType::Data;
  scale_ = layer_param_.data_param().scale();
  mean_value_ = layer_param_.data_param().mean_value();

  int batch = shape->dim(0);
  int in_c = shape->dim(1), in_h = shape->dim(2), in_w = shape->dim(3);

  int num = in_c * in_h * in_w;

  *in_blob->mutable_shape() = *shape;
  *out_blob->mutable_shape() = *shape;

  in_blob->set_num(num);
  out_blob->set_num(num);
  in_blob->set_count(batch * num);
  out_blob->set_count(batch * num);

  out_data_ = new float[out_blob->count()];

#ifdef USE_CUDA
  cuda_in_data_ = CUDA::CUDAMakeBuffer(in_blob->count(), NULL);
  cuda_out_data_ = CUDA::CUDAMakeBuffer(out_blob->count(), NULL);
#endif

#ifdef USE_CL
  cl_in_data_ = CL::CLMakeBuffer(in_blob->count(), CL_MEM_READ_WRITE, NULL);
  cl_out_data_ = CL::CLMakeBuffer(out_blob->count(), CL_MEM_READ_WRITE, NULL);
#endif

#ifdef VERBOSE
  printf("Data Layer: %d x %d x %d input\n", in_c, in_h, in_w);
#endif
}

void DataLayer::ForwardLayer(float *in_data) {
  for (int i = 0; i < in_blob->count(); ++i) {
    out_data_[i] = (in_data[i] - mean_value_) * scale_;
  }
}

#ifdef USE_CUDA
void DataLayer::CUDAForwardLayer(float *in_data) {
  in_data_ = in_data;
  CUDA::CUDAWriteBuffer(in_blob->count(), cuda_in_data_, in_data_);
  Kernel::CUDADataTransform(out_blob->count(), cuda_in_data_, scale_,
                            mean_value_, cuda_out_data_);
}
#endif

#ifdef USE_CL
void DataLayer::CLForwardLayer(float *in_data) {
  in_data_ = in_data;
  CL::CLWriteBuffer(in_blob->count(), cl_in_data_, in_data_);
  Kernel::CLDataTransform(out_blob->count(), cl_in_data_, scale_, mean_value_,
                          cl_out_data_);
}
#endif

void DataLayer::ReleaseLayer() {
  if (out_data_ != NULL)
    delete[] out_data_;

#ifdef USE_CUDA
  if (cuda_in_data_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_in_data_);
  if (cuda_out_data_ != NULL)
    CUDA::CUDAReleaseBuffer(cuda_out_data_);
#endif

#ifdef USE_CL
  if (cl_in_data_ != NULL)
    CL::CLReleaseBuffer(cl_in_data_);
  if (cl_out_data_ != NULL)
    CL::CLReleaseBuffer(cl_out_data_);
#endif
  // std::cout << "Free DataLayer!" << std::endl;
}
