#include "binary_op.hpp"

namespace Shadow {

void BinaryOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  bottom_shape_ = bottom->shape();

  BlobF *scalar = nullptr;
  if (has_scalar_arg_) {
    top_shape_ = bottom_shape_;
  } else {
    CHECK_EQ(bottoms_size(), 2);
    scalar = const_cast<BlobF *>(bottoms<float>(1));
    scalar_shape_ = scalar->shape();
    need_broadcast_ = bottom_shape_ != scalar_shape_;
    if (need_broadcast_) {
      int bottom_num_axes = bottom->num_axes();
      int scalar_num_axes = scalar->num_axes();
      if (bottom_num_axes > scalar_num_axes) {
        for (int n = 0; n < bottom_num_axes - scalar_num_axes; ++n) {
          scalar_shape_.insert(scalar_shape_.begin(), 1);
        }
      } else {
        for (int n = 0; n < scalar_num_axes - bottom_num_axes; ++n) {
          bottom_shape_.insert(bottom_shape_.begin(), 1);
        }
      }
      CHECK_EQ(bottom_shape_.size(), scalar_shape_.size());
      top_shape_.clear();
      for (int n = 0; n < bottom_shape_.size(); ++n) {
        int bottom_dim = bottom_shape_[n], scalar_dim = scalar_shape_[n];
        CHECK(bottom_dim == scalar_dim || bottom_dim == 1 || scalar_dim == 1);
        top_shape_.push_back(std::max(bottom_dim, scalar_dim));
      }
    } else {
      top_shape_ = bottom_shape_;
    }
  }

  if (bottom != top && scalar != top) {
    top->reshape(top_shape_);
  }

  if (!has_scalar_arg_ && need_broadcast_) {
    int num_axes = static_cast<int>(top_shape_.size());

    op_ws_->GrowTempBuffer(3 * num_axes, sizeof(int));

    auto *bottom_shape =
        op_ws_->CreateTempBlob<int>({num_axes}, op_name_ + "/bottom_shape");
    auto *scalar_shape =
        op_ws_->CreateTempBlob<int>({num_axes}, op_name_ + "/scalar_shape");
    auto *top_shape =
        op_ws_->CreateTempBlob<int>({num_axes}, op_name_ + "/top_shape");

    bottom_shape->set_data(bottom_shape_.data(), num_axes);
    scalar_shape->set_data(scalar_shape_.data(), num_axes);
    top_shape->set_data(top_shape_.data(), num_axes);

    return Vision::BroadcastBinary(bottom->data(), bottom_shape->data(),
                                   scalar->data(), scalar_shape->data(),
                                   operation_, num_axes, top->count(),
                                   top_shape->data(), top->mutable_data());
  }

  int count = top->count();

  switch (operation_) {
    case kAdd:
      if (has_scalar_arg_) {
        return Blas::Add(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Add(count, bottom->data(), 0, scalar->data(), 0,
                         top->mutable_data(), 0);
      }
    case kSub:
      if (has_scalar_arg_) {
        return Blas::Sub(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Sub(count, bottom->data(), 0, scalar->data(), 0,
                         top->mutable_data(), 0);
      }
    case kMul:
      if (has_scalar_arg_) {
        return Blas::Mul(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Mul(count, bottom->data(), 0, scalar->data(), 0,
                         top->mutable_data(), 0);
      }
    case kDiv:
      if (has_scalar_arg_) {
        return Blas::Div(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Div(count, bottom->data(), 0, scalar->data(), 0,
                         top->mutable_data(), 0);
      }
    case kPow:
      if (has_scalar_arg_) {
        return Blas::Pow(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Pow(count, bottom->data(), 0, scalar->data(), 0,
                         top->mutable_data(), 0);
      }
    case kMax:
      if (has_scalar_arg_) {
        return Blas::Max(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Max(count, bottom->data(), 0, scalar->data(), 0,
                         top->mutable_data(), 0);
      }
    case kMin:
      if (has_scalar_arg_) {
        return Blas::Min(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Min(count, bottom->data(), 0, scalar->data(), 0,
                         top->mutable_data(), 0);
      }
    default:
      LOG(FATAL) << "Unknown binary operation " << operation_;
  }
}

REGISTER_OPERATOR(Binary, BinaryOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
inline T Binary(T a, T b, int operation) {
  switch (operation) {
    case 0:
      return a + b;
    case 1:
      return a - b;
    case 2:
      return a * b;
    case 3:
      return a / b;
    case 4:
      return std::pow(a, b);
    case 5:
      return std::max(a, b);
    case 6:
      return std::min(a, b);
    default:
      return 0;
  }
}

template <typename T>
void BroadcastBinary(const T *in_data, const int *in_shape,
                     const T *scalar_data, const int *scalar_shape,
                     int operation, int num_axes, int count,
                     const int *out_shape, T *out_data) {
  VecInt in_shape_acc(1, 1), scalar_shape_acc(1, 1);
  for (int n = num_axes - 1; n > 0; --n) {
    in_shape_acc.insert(in_shape_acc.begin(), in_shape[n] * in_shape_acc[0]);
    scalar_shape_acc.insert(scalar_shape_acc.begin(),
                            scalar_shape[n] * scalar_shape_acc[0]);
  }
  for (int i = 0; i < count; ++i) {
    int in_index = 0, scalar_index = 0, cc = i;
    for (int n = num_axes - 1; n >= 0; --n) {
      int dim = cc % out_shape[n];
      in_index += (dim % in_shape[n]) * in_shape_acc[n];
      scalar_index += (dim % scalar_shape[n]) * scalar_shape_acc[n];
      cc /= out_shape[n];
    }
    out_data[i] =
        Binary(in_data[in_index], scalar_data[scalar_index], operation);
  }
}

template void BroadcastBinary(const float *in_data, const int *in_shape,
                              const float *scalar_data, const int *scalar_shape,
                              int operation, int num_axes, int count,
                              const int *out_shape, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
