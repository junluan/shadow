#include "data_op.hpp"

namespace Shadow {

void DataOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  top->reshape(bottom->shape());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void DataOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::DataTransform(bottom->data(), bottom->shape(), num_mean_,
                        mean_value_->data(), num_scale_, scale_value_->data(),
                        top->mutable_data());
}

REGISTER_OPERATOR(Data, DataOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, int num_mean,
                   const T *mean_value, int num_scale, const T *scale_value,
                   T *out_data) {
  int in_c = in_shape[1], spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_c * spatial_dim;
  if (num_mean == 1 && num_scale == 1) {
    for (int i = 0; i < count; ++i) {
      out_data[i] = (in_data[i] - mean_value[0]) * scale_value[0];
    }
  } else if (num_mean == in_c && num_scale == 1) {
    for (int i = 0; i < count; ++i) {
      int c_out = (i / spatial_dim) % in_c;
      out_data[i] = (in_data[i] - mean_value[c_out]) * scale_value[0];
    }
  } else if (num_mean == 1 && num_scale == in_c) {
    for (int i = 0; i < count; ++i) {
      int c_out = (i / spatial_dim) % in_c;
      out_data[i] = (in_data[i] - mean_value[0]) * scale_value[c_out];
    }
  } else if (num_mean == in_c && num_scale == in_c) {
    for (int i = 0; i < count; ++i) {
      int c_out = (i / spatial_dim) % in_c;
      out_data[i] = (in_data[i] - mean_value[c_out]) * scale_value[c_out];
    }
  } else {
    LOG(FATAL) << "Number of mean or scale must be one or the same with "
                  "channel number, current is mean: "
               << num_mean << ", scale: " << num_scale;
  }
}

template void DataTransform(const float *in_data, const VecInt &in_shape,
                            int num_mean, const float *mean_value,
                            int num_scale, const float *scale_value,
                            float *out_data);

#elif defined(USE_CL)
template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, int num_mean,
                   const T *mean_value, int num_scale, const T *scale_value,
                   T *out_data) {
  int in_c = in_shape[1], spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_c * spatial_dim;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["DataTransform"];
  kernel->SetArguments(*in_data, count, in_c, spatial_dim, num_mean,
                       *mean_value, num_scale, *scale_value, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void DataTransform(const BufferF *in_data, const VecInt &in_shape,
                            int num_mean, const BufferF *mean_value,
                            int num_scale, const BufferF *scale_value,
                            BufferF *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
