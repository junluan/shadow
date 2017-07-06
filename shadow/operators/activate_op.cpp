#include "activate_op.hpp"
#include "core/blas.hpp"
#include "core/image.hpp"

namespace Shadow {

void ActivateOp::Setup() {
  activate_type_ = arg_helper_.GetSingleArgument<int>("type", 1);

  if (activate_type_ == 3) {
    channel_shared_ =
        arg_helper_.GetSingleArgument<bool>("channel_shared", false);
    CHECK_GE(bottoms_[0]->num_axes(), 2);
    int channels = bottoms_[0]->shape(1);
    if (blobs_.size() == 0) {
      if (channel_shared_) {
        blobs_.push_back(new BlobF(VecInt(1, 1)));
      } else {
        blobs_.push_back(new BlobF(VecInt(1, channels)));
      }
      Blas::Set(blobs_[0]->count(), 0.25, blobs_[0]->mutable_data(), 0);
    }
    if (channel_shared_) {
      CHECK_EQ(blobs_[0]->count(), 1);
    } else {
      CHECK_EQ(blobs_[0]->count(), channels);
    }
  }
}

void ActivateOp::Reshape() {
  if (bottoms_[0] != tops_[0]) {
    tops_[0]->reshape(bottoms_[0]->shape());
  }

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ActivateOp::Forward() {
  if (bottoms_[0] != tops_[0]) {
    Blas::BlasScopy(bottoms_[0]->count(), bottoms_[0]->data(), 0,
                    tops_[0]->mutable_data(), 0);
  }
  if (activate_type_ == 3) {
    Image::PRelu(tops_[0]->mutable_data(), tops_[0]->shape(), channel_shared_,
                 blobs_[0]->data());
  } else {
    Image::Activate(tops_[0]->mutable_data(), tops_[0]->count(),
                    activate_type_);
  }
}

void ActivateOp::Release() {
  // DLOG(INFO) << "Free ActivateOp!";
}

REGISTER_OPERATOR(Activate, ActivateOp);

}  // namespace Shadow
