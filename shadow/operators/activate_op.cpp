#include "activate_op.hpp"

namespace Shadow {

void ActivateOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0);
  }

  // PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
  if (activate_type_ == kRelu || activate_type_ == kLeaky ||
      activate_type_ == kSigmoid || activate_type_ == kSoftPlus ||
      activate_type_ == kTanh) {
    Vision::Activate(top->mutable_data(), top->count(), activate_type_, slope_);
  } else if (activate_type_ == kPRelu) {
    CHECK_EQ(bottoms_size(), 2);
    CHECK_GE(bottom->num_axes(), 2);
    const auto *slope = bottoms<float>(1);
    if (channel_shared_) {
      CHECK_EQ(slope->count(), 1);
    } else {
      CHECK_EQ(slope->count(), bottom->shape(1));
    }
    Vision::PRelu(top->mutable_data(), top->shape(), channel_shared_,
                  slope->data());
  }

  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(Activate, ActivateOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
inline T Activate(T x, int type, float slope) {
  switch (type) {
    case 1:
      return x * (x > 0);
    case 2:
      return x > 0 ? x : T(slope * x);
    case 3:
      return 1 / (1 + std::exp(-x));
    case 4:
      return std::log(1 + std::exp(x));
    case 5: {
      T exp_2x = std::exp(2 * x);
      return (exp_2x - 1) / (exp_2x + 1);
    }
    default:
      return x;
  }
}

template <typename T>
void Activate(T *data, int count, int type, float slope) {
// PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
#if defined(USE_Eigen)
  auto data_eigen = MapVector<T>(data, count);
  switch (type) {
    case 1:
      data_eigen = data_eigen.cwiseMax(T(0));
      break;
    case 2:
      data_eigen = data_eigen.unaryExpr(
          [slope](T x) { return x > 0 ? x : T(slope * x); });
      break;
    case 3:
      data_eigen =
          data_eigen.unaryExpr([](T x) { return 1 / (1 + std::exp(-x)); });
      break;
    case 4:
      data_eigen =
          data_eigen.unaryExpr([](T x) { return std::log(1 + std::exp(x)); });
      break;
    case 5:
      data_eigen = data_eigen.unaryExpr([](T x) {
        T exp_2x = std::exp(2 * x);
        return (exp_2x - 1) / (exp_2x + 1);
      });
      break;
    default:
      return;
  }
#else
  for (int i = 0; i < count; ++i) {
    data[i] = Activate(data[i], type, slope);
  }
#endif
}

template <typename T>
void PRelu(T *data, const VecInt &in_shape, bool channel_shared,
           const T *slope_data) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    data[i] = data[i] > 0 ? data[i] : data[i] * slope_data[c];
  }
}

template void Activate(float *data, int count, int type, float slope);
template void PRelu(float *data, const VecInt &in_shape, bool channel_shared,
                    const float *slope_data);

#elif defined(USE_CL)
template <typename T>
void Activate(T *data, int count, int type, float slope) {
  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Activate"];
  kernel->SetArguments(*data, count, type, slope);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void PRelu(T *data, const VecInt &in_shape, bool channel_shared,
           const T *slope_data) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["PRelu"];
  kernel->SetArguments(*data, count, channels, dim, div_factor, *slope_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Activate(BufferF *data, int count, int type, float slope);
template void PRelu(BufferF *data, const VecInt &in_shape, bool channel_shared,
                    const BufferF *slope_data);
#endif

}  // namespace Vision

}  // namespace Shadow
