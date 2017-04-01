#include "shadow/layers/activate_layer.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

void ActivateLayer::Setup(VecBlobF *blobs) {
  Layer::Setup(blobs);

  const auto &activate_param = layer_param_.activate_param();

  activate_type_ = activate_param.type();

  if (activate_type_ == shadow::ActivateParameter_ActivateType_PRelu) {
    channel_shared_ = activate_param.channel_shared();
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

void ActivateLayer::Reshape() {
  if (bottoms_[0] != tops_[0]) {
    tops_[0]->reshape(bottoms_[0]->shape());
  }

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ActivateLayer::Forward() {
  if (bottoms_[0] != tops_[0]) {
    Blas::BlasScopy(bottoms_[0]->count(), bottoms_[0]->data(), 0,
                    tops_[0]->mutable_data(), 0);
  }
  if (activate_type_ == shadow::ActivateParameter_ActivateType_PRelu) {
    Image::PRelu(tops_[0]->mutable_data(), tops_[0]->shape(), channel_shared_,
                 blobs_[0]->data());
  } else {
    Image::Activate(tops_[0]->mutable_data(), tops_[0]->count(),
                    activate_type_);
  }
}

void ActivateLayer::Release() {
  // DLOG(INFO) << "Free ActivateLayer!";
}
