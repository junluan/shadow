#include "activate_op.hpp"

namespace Shadow {

void ActivateOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    int batch = bottom->shape(0), num = bottom->num();

    cudnn::setActivationDesc<float>(&activate_desc_, activate_type_,
                                    static_cast<double>(slope_));
    cudnn::setTensor4dDesc<float>(&bottom_top_desc_, batch, num, 1, 1);

    CUDNN_CHECK(cudnnActivationForward(
        cudnnHandle_t(op_ws_->Ctx()->cudnn_handle()), activate_desc_,
        cudnn::dataType<float>::one, bottom_top_desc_, bottom->data(),
        cudnn::dataType<float>::zero, bottom_top_desc_, top->mutable_data()));

    return;
  }
#endif

  if (activate_type_ == kRelu || activate_type_ == kLeaky ||
      activate_type_ == kSigmoid || activate_type_ == kSoftPlus ||
      activate_type_ == kTanh || activate_type_ == kRelu6) {
    Vision::Activate(bottom->data(), top->mutable_data(), top->count(),
                     activate_type_, slope_);
  } else if (activate_type_ == kPRelu) {
    CHECK_EQ(bottoms_size(), 2);
    CHECK_GE(bottom->num_axes(), 2);
    const auto *slope = bottoms<float>(1);
    bool channel_shared = slope->count() == 1;
    if (!channel_shared) {
      CHECK_EQ(slope->count(), bottom->shape(1));
    }
    Vision::PRelu(bottom->data(), top->mutable_data(), top->shape(),
                  channel_shared, slope->data());
  }
}

REGISTER_OPERATOR(Activate, ActivateOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
inline T Activate(T x, int type, float slope) {
  switch (type) {
    case ActivateOp::kRelu:
      return x > 0 ? x : 0;
    case ActivateOp::kLeaky:
      return x > 0 ? x : T(slope * x);
    case ActivateOp::kSigmoid:
      return 1 / (1 + std::exp(-x));
    case ActivateOp::kSoftPlus:
      return std::log(1 + std::exp(x));
    case ActivateOp::kTanh: {
      T exp_2x = std::exp(2 * x);
      return (exp_2x - 1) / (exp_2x + 1);
    }
    case ActivateOp::kRelu6: {
      x = x > 0 ? x : 0;
      return x < 6 ? x : 6;
    }
    default:
      return x;
  }
}

template <typename T>
void Activate(const T *in_data, T *out_data, int count, int type, float slope) {
#if defined(USE_Eigen)
  const auto &in_eigen = MapVector<T>(const_cast<T *>(in_data), count);
  auto out_eigen = MapVector<T>(out_data, count);
  switch (type) {
    case ActivateOp::kRelu:
      out_eigen = in_eigen.cwiseMax(T(0));
      break;
    case ActivateOp::kLeaky:
      out_eigen =
          in_eigen.unaryExpr([slope](T x) { return x > 0 ? x : T(slope * x); });
      break;
    case ActivateOp::kSigmoid:
      out_eigen =
          in_eigen.unaryExpr([](T x) { return 1 / (1 + std::exp(-x)); });
      break;
    case ActivateOp::kSoftPlus:
      out_eigen =
          in_eigen.unaryExpr([](T x) { return std::log(1 + std::exp(x)); });
      break;
    case ActivateOp::kTanh:
      out_eigen = in_eigen.unaryExpr([](T x) {
        T exp_2x = std::exp(2 * x);
        return (exp_2x - 1) / (exp_2x + 1);
      });
      break;
    case ActivateOp::kRelu6:
      out_eigen = in_eigen.cwiseMax(T(0)).cwiseMin(T(6));
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

template void Activate(const float *, float *, int, int, float);
template void PRelu(const float *, float *, const VecInt &, bool,
                    const float *);
#endif

}  // namespace Vision

}  // namespace Shadow
