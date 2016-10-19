#include "shadow/layers/activate_layer.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

void ActivateLayer::Reshape() {
  activate_type_ = layer_param_.activate_param().type();

  if (bottom_[0] != top_[0]) {
    top_[0]->reshape(bottom_[0]->shape());
  }

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ActivateLayer::Forward() {
  if (bottom_[0] != top_[0]) {
    Blas::BlasScopy(bottom_[0]->count(), bottom_[0]->data(), 0,
                    top_[0]->mutable_data(), 0);
  }
  Image::Activate(top_[0]->mutable_data(), top_[0]->count(), activate_type_);
}

void ActivateLayer::Release() {
  bottom_.clear();
  top_.clear();

  // DInfo("Free ActivateLayer!");
}
