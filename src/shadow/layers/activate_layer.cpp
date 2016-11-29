#include "shadow/layers/activate_layer.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

void ActivateLayer::Setup(VecBlob *blobs) {
  Layer::Setup(blobs);

  const auto &activate_param = layer_param_.activate_param();

  activate_type_ = activate_param.type();
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
  Image::Activate(tops_[0]->mutable_data(), tops_[0]->count(), activate_type_);
}

void ActivateLayer::Release() {
  // DLOG(INFO) << "Free ActivateLayer!";
}
