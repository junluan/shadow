#include "activate_op.hpp"
#include "core/blas.hpp"
#include "core/image.hpp"

namespace Shadow {

void ActivateOp::Setup() {
  activate_type_ = get_single_argument<int>("type", 1);
  CHECK_GE(activate_type_, 0);
  CHECK_LE(activate_type_, 3);

  if (activate_type_ == 3) {
    channel_shared_ = get_single_argument<bool>("channel_shared", false);
    CHECK_GE(bottoms<float>(0)->num_axes(), 2);
    int channels = bottoms<float>(0)->shape(1);
    if (blobs_size() == 0) {
      add_blobs<float>(op_name_ + "_param");
      auto *param_blob = mutable_blobs<float>(0);
      if (channel_shared_) {
        param_blob->reshape({1, 1});
      } else {
        param_blob->reshape({1, channels});
      }
      Blas::Set(param_blob->count(), 0.25, param_blob->mutable_data(), 0);
    }
    if (channel_shared_) {
      CHECK_EQ(blobs<float>(0)->count(), 1);
    } else {
      CHECK_EQ(blobs<float>(0)->count(), channels);
    }
  }
}

void ActivateOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> t"
             << activate_type_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void ActivateOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0);
  }

  // Linear: 0, Relu: 1, Leaky: 2, PRelu: 3
  if (activate_type_ == kRelu || activate_type_ == kLeaky) {
    Image::Activate(top->mutable_data(), top->count(), activate_type_);
  } else if (activate_type_ == kPRelu) {
    Image::PRelu(top->mutable_data(), top->shape(), channel_shared_,
                 blobs<float>(0)->data());
  }
}

void ActivateOp::Release() {
  // DLOG(INFO) << "Free ActivateOp!";
}

REGISTER_OPERATOR(Activate, ActivateOp);

}  // namespace Shadow
