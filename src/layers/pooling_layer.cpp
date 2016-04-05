#include "pooling_layer.h"
#include "kernel.h"

PoolingLayer::PoolingLayer(LayerType type) { layer_type_ = type; }
PoolingLayer::~PoolingLayer() {}

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
  int h_offset = ((in_h_ - ksize_) % stride_) / 2;
  int w_offset = ((in_w_ - ksize_) % stride_) / 2;

  for (int b = 0; b < batch_; ++b) {
    for (int c = 0; c < out_c_; ++c) {
      for (int h = 0; h < out_h_; ++h) {
        for (int w = 0; w < out_w_; ++w) {
          int out_index = w + out_w_ * (h + out_h_ * (c + out_c_ * b));
          float max = -10000.0f;
          float sum = 0;
          for (int ki = 0; ki < ksize_; ++ki) {
            for (int kj = 0; kj < ksize_; ++kj) {
              int cur_h = h_offset + h * stride_ + ki;
              int cur_w = w_offset + w * stride_ + kj;
              int index = cur_w + in_w_ * (cur_h + in_h_ * (c + b * in_c_));
              bool valid =
                  (cur_h >= 0 && cur_h < in_h_ && cur_w >= 0 && cur_w < in_w_);
              float value = valid ? in_data_[index] : -10000.0f;
              max = (value > max) ? value : max;
              sum += valid ? in_data_[index] : 0.;
            }
          }
          if (pool_type_ == kMax)
            out_data_[out_index] = max;
          else
            out_data_[out_index] = sum / (ksize_ * ksize_);
        }
      }
    }
  }
}

#ifdef USE_CL
void PoolingLayer::CLForwardLayer() {
  Kernel::CLPooling(cl_in_data_, batch_, in_c_, in_h_, in_w_, ksize_, stride_,
                    out_h_, out_w_, pool_type_, cl_out_data_);
}
#endif

void PoolingLayer::ReleaseLayer() {
  if (out_data_ != NULL)
    delete[] out_data_;

#ifdef USE_CL
  if (cl_out_data_ != NULL)
    CL::CLReleaseBuffer(cl_out_data_);
#endif
  // std::cout << "Free PoolingLayer!" << std::endl;
}
