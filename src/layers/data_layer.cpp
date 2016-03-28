#include "data_layer.h"

DataLayer::DataLayer(LayerType type) { layer_type_ = type; }
DataLayer::~DataLayer() {}

void DataLayer::MakeDataLayer(SizeParams params) {
  batch_ = params.batch;
  in_c_ = params.in_c;
  in_h_ = params.in_h;
  in_w_ = params.in_w;
  out_c_ = in_c_;
  out_h_ = in_h_;
  out_w_ = in_w_;

  in_num_ = in_c_ * in_h_ * in_w_;
  out_num_ = out_c_ * out_h_ * out_w_;

  out_data_ = new float[batch_ * out_num_];

#ifdef USE_CL
  cl_in_data_ = CL::CLMakeBuffer(batch_ * in_num_, CL_MEM_READ_WRITE, NULL);
  cl_out_data_ = CL::CLMakeBuffer(batch_ * out_num_, CL_MEM_READ_WRITE, NULL);
#endif

#ifdef VERBOSE
  printf("Data Layer: %d x %d x %d input\n", in_h_, in_w_, in_c_);
#endif
}

void DataLayer::ForwardLayer(float *in_data) {
  for (int i = 0; i < batch_ * in_num_; ++i) {
    out_data_[i] = (in_data[i] - mean_value_) * scale_;
  }
}

#ifdef USE_CL
void DataLayer::CLForwardLayer(float *in_data) {
  in_data_ = in_data;
  CL::CLWriteBuffer(batch_ * in_num_, cl_in_data_, in_data_);
  CL::CLDataTransform(batch_ * out_num_, cl_in_data_, scale_, mean_value_,
                      cl_out_data_);
}
#endif

void DataLayer::ReleaseLayer() {
  if (in_data_ != NULL)
    delete[] in_data_;
  if (out_data_ != NULL)
    delete[] out_data_;

#ifdef USE_CL
  if (cl_in_data_ != NULL)
    CL::CLReleaseBuffer(cl_in_data_);
  if (cl_out_data_ != NULL)
    CL::CLReleaseBuffer(cl_out_data_);
#endif
  // std::cout << "Free DataLayer!" << std::endl;
}
