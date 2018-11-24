#include "activate_op.hpp"

namespace Shadow {

void ActivateOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  // PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
  if (activate_type_ == kRelu || activate_type_ == kLeaky ||
      activate_type_ == kSigmoid || activate_type_ == kSoftPlus ||
      activate_type_ == kTanh) {
    Vision::Activate(bottom->data(), top->mutable_data(), top->count(),
                     activate_type_, slope_);
  } else if (activate_type_ == kPRelu) {
    CHECK_EQ(bottoms_size(), 2);
    CHECK_GE(bottom->num_axes(), 2);
    const auto *slope = bottoms<float>(1);
    if (channel_shared_) {
      CHECK_EQ(slope->count(), 1);
    } else {
      CHECK_EQ(slope->count(), bottom->shape(1));
    }
    Vision::PRelu(bottom->data(), top->mutable_data(), top->shape(),
                  channel_shared_, slope->data());
  }
}

REGISTER_OPERATOR(Activate, ActivateOp);

namespace Vision {

#if !defined(USE_CUDA)
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
void Activate(const T *in_data, T *out_data, int count, int type, float slope) {
// PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
#if defined(USE_Eigen)
  const auto &in_eigen = MapVector<T>(const_cast<T *>(in_data), count);
  auto out_eigen = MapVector<T>(out_data, count);
  switch (type) {
    case 1:
      out_eigen = in_eigen.cwiseMax(T(0));
      break;
    case 2:
      out_eigen =
          in_eigen.unaryExpr([slope](T x) { return x > 0 ? x : T(slope * x); });
      break;
    case 3:
      out_eigen =
          in_eigen.unaryExpr([](T x) { return 1 / (1 + std::exp(-x)); });
      break;
    case 4:
      out_eigen =
          in_eigen.unaryExpr([](T x) { return std::log(1 + std::exp(x)); });
      break;
    case 5:
      out_eigen = in_eigen.unaryExpr([](T x) {
        T exp_2x = std::exp(2 * x);
        return (exp_2x - 1) / (exp_2x + 1);
      });
      break;
    default:
      return;
  }
#else
  for (int i = 0; i < count; ++i) {
    out_data[i] = Activate(in_data[i], type, slope);
  }
#endif
}

template <typename T>
void PRelu(const T *in_data, T *out_data, const VecInt &in_shape,
           bool channel_shared, const T *slope_data) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    out_data[i] = in_data[i] > 0 ? in_data[i] : in_data[i] * slope_data[c];
  }
}

template void Activate(const float *in_data, float *out_data, int count,
                       int type, float slope);
template void PRelu(const float *in_data, float *out_data,
                    const VecInt &in_shape, bool channel_shared,
                    const float *slope_data);
#endif

}  // namespace Vision

}  // namespace Shadow
